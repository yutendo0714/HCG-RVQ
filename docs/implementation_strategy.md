# HCG-RVQ Implementation Strategy

Date: 2026-05-29

## Current Position

The core HCG-RVQ proposal should remain self-implemented in this repository. The central claim is not just "use RVQ in LIC"; it is that the hyperprior can generate the local quantizer geometry. That claim has to be isolated in code, ablations, and attribution.

Current custom modules:

- `hcg_rvq/models/hyperprior_rvq.py`: MeanScaleHyperprior backbone plus Global RVQ, HCS-RVQ, and HCG-RVQ-H integration.
- `hcg_rvq/quantizers.py`: residual vector quantization and Householder transform.
- `hcg_rvq/entropy.py`: VQ index entropy model.
- `tools/init_rvq_codebook.py`: scalar-checkpoint latent sampling and RVQ codebook initialization.

This is the right default for novelty. Importing a full external VQ-LIC method into the core path too early would make it hard to say whether gains come from HCG geometry or from a stronger external recipe.

## Where Official Implementations Should Be Used

Use official or near-official implementations aggressively for support and comparison, but not as a black-box replacement for the proposed module.

### Keep Using as Infrastructure

- CompressAI: entropy bottleneck, Gaussian conditional, baseline model conventions, evaluation patterns, and pretrained-reference comparisons.

### Use as Baseline or Reproduction Targets

- RDVQ: primary VQ-based compression competitor and reference for differentiable RD optimization of VQ distributions.
- MLIC / MLIC++: strong context and entropy-model baseline.
- HPCM: progressive context modeling; relevant to future RVQ stage-context entropy modeling.
- MambaIC: strong modern backbone candidate after the MeanScaleHyperprior ablations are clean.
- DCAE: dictionary-based entropy-model competitor and discussion point.

### Use as Stabilization References, Not Drop-In Core Code

- SimVQ: codebook collapse mitigation through code-vector reparameterization.
- vector-quantize-pytorch: practical VQ/RVQ/FSQ implementation patterns, EMA options, and utilization diagnostics.

## Recommended Policy

1. Keep the HCG quantizer path self-contained until the core ablations are convincing.
2. Borrow engineering ideas only after recording a matched ablation: initialization, EMA, SimVQ-style reparameterization, rotation trick, or FSQ fallback.
3. Do not copy external code into the core path without checking license, citation, and whether the change confounds the novelty claim.
4. For SOTA comparison, prefer running official repositories or reporting official numbers under clearly stated protocols rather than reimplementing large backbones in this repo.
5. Integrate HCG into one strong backbone only after the simple-backbone story is stable: Global RVQ -> HCS -> HCG-H -> index entropy -> stage context/gating.

## Immediate Research Actions

1. Use `tools/evaluate_checkpoints.py` for validation-selected checkpointing because both HCS and HCG-H currently peak around 500 pilot steps and then lose Kodak generalization.
2. Run frozen 500-step HCS/HCG-H across three seeds before claiming a geometry gain.
3. Use the completed no-geometry matched control as the attribution baseline for any Householder claim.
4. Add conditioning-drift regularization before unfreezing `h_s` again.
5. If codebook utilization weakens, test SimVQ-style codebook reparameterization as a separate ablation.
6. After the minimal evidence is stable, compare against official RDVQ and strong LIC baselines, then consider a plugin experiment on MLIC/HPCM/MambaIC.

## EF-LIC Local HCG Head Plan After E221-E223

The next EF-LIC implementation should move from post-hoc selector diagnostics to an in-codec local HCG head. The insertion point is immediately after `mean, scale = _mean_scale(support_buf, i)` and before `quantizes[force_ind][i]` receives the normalized y-slice. The decoder can reproduce the same decision at the matching point in `decompress`, because it has the same z indices, decoded z, hyper support, previous decoded slices, and current slice mean/scale. The head must not use raw `y_norm` or active-vs-baseline residual labels at inference time.

Recommended first implementation unit:

- Keep EF-LIC weights and existing RVQ codebooks frozen for the first pilot.
- Add a small decoder-safe local head that consumes current local `mean`, `scale`, the available prefix of `support_buf`, slice id, and z/index prior summaries or broadcast z-prior maps.
- Output an `alpha` map or small set of alpha logits over `{0, 0.01, 0.02}`. `alpha=0` is the explicit fallback.
- Use the existing projected-HCG stage implementation as the active geometry path, but replace fixed scalar `slice_alphas` with the head output.
- Train first on independent teacher labels from E221-style quant-MSE or E160/E213-style metric oracles with a false-positive-heavy loss. Then move to codec-aware fine-tuning only after encoder/decode equality and held-out transfer are stable.

Why this is now justified:

- E221 shows strong local oracle headroom at unchanged index rate.
- E222 shows raw decoder-safe linear features can exploit some same-table signal but fail transfer.
- E223 shows image/slice normalization and dataset-balanced thresholds do not rescue transfer.

Therefore the next paper-facing method should not be a hand-written threshold. It should be a learned hyperprior-conditioned geometry/strength module whose ablations include fixed alpha, fixed slice schedule, post-hoc local ridge, learned local head, and oracle upper bounds. Full evaluation should then return to Kodak/Tecnick/CLIC/OpenImages with paper-style RD and perceptual metrics.

## EF-LIC Local Head Update After E236-E238

The local head should be trained as a confidence-gated family/strength
controller rather than as an all-on geometry module.

Updated implementation target:

- Insert after `_mean_scale(support_buf, i)` in EF-LIC and mirror the same
  computation in `decompress`.
- Input local decoder-safe maps: current slice `mean`, `scale`, slice id,
  decoded prefix/support features, and optional z-prior maps or broadcast z
  summaries.
- Output `zero/fallback` plus coarse HCG family/strength logits. First families
  should mirror the E236 vocabulary: constant, guarded constant, guarded
  support, soft blend, sparse union, hybrid, and zero.
- Train first with high-confidence E238 labels: nonzero activation only when
  oracle gain and family margin exceed conservative thresholds; otherwise zero.
- Use an asymmetric loss: wrong nonzero activation on true-zero or low-margin
  labels is more costly than falling back on small gains.
- Keep the first pilot frozen except for the head. Only after encoder/decode
  equality and held-out false-positive rates are stable should the geometry path
  or EF-LIC backbone be fine-tuned.

Rationale:

- E236: local codec-valid basis states have large oracle headroom.
- E237: global image-level selectors fail, but true-family selection nearly
  recovers the oracle.
- E238: high-confidence labels retain about 91% of pooled oracle headroom while
  activating only about 74% of images, so a conservative learned head is a
  realistic next paper-main implementation.

## EF-LIC E239 Trainable Head Contract

The first implementation target is now explicit:

- `hcg_rvq/eflic_local_controller.py` defines the decoder-safe local context
  maps, `LocalHCGFamilyHead`, and an asymmetric family loss.
- `tools/build_e239_eflic_local_head_training_plan.py` converts E238 labels and
  family costs into a manifest, class weights, and cost matrix.
- The initial head should be trained with EF-LIC and HCG basis states frozen.
  Use image-level labels as a broadcast smoke target only to validate the
  contract, then move to slice/spatial labels.
- Keep zero/fallback as the default state. The E239 manifest retains about
  91% of oracle headroom while activating 74% of images, so forcing activation
  everywhere is unnecessary and empirically risky.
- Treat hybrid as disabled or diagnostic in the first supervised run unless a
  later spatial-label pass produces high-confidence hybrid targets.

Recommended next experiment:

1. Train `LocalHCGFamilyHead` from the E239 manifest with frozen EF-LIC/HCG.
2. Verify forward/decompress equality and finite metrics on GPU0 only.
3. Evaluate Kodak24 and CLIC professional with per-image family predictions,
   zero activation rate, false-positive rate, and RD/perceptual deltas.
4. Promote to spatial/slice labels or strength regression only if held-out
   false positives stay controlled.

## EF-LIC E240 Context Export Contract

The local-head training data path is now validated on a real EF-LIC forward
pass. Use `tools/export_e240_eflic_local_head_contexts.py` to export tensors
with shape `[4, 11, H, W]` and E239 targets.

Recommended commands:

1. Kodak24 export:
   `CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/export_e240_eflic_local_head_contexts.py --image-dir experiments/data/kodak24 --dataset kodak24 --max-images 24 --device cuda:0 --output-dir experiments/analysis/e240_eflic_local_head_contexts_kodak24`
2. CLIC professional export after placing or locating the CLIC professional images:
   `CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/export_e240_eflic_local_head_contexts.py --image-dir <clic-professional-image-dir> --dataset clicpro41 --max-images 41 --device cuda:0 --output-dir experiments/analysis/e240_eflic_local_head_contexts_clicpro41`

Do not treat these exports as final labels. They are the frozen-head smoke
bridge. The next code unit should train `LocalHCGFamilyHead` from these tensors
with E239 class weights and cost matrix, then report family confusion,
zero-activation rate, false-positive nonzero rate, and held-out RD impact.

## EF-LIC E241 Frozen Head Result and E242 Direction

The full Kodak24 E240 context export is now complete and finite, but E241 shows
that E239 image-level labels should remain a smoke target only. Broadcast
training collapses to nonzero predictions, and image-level pooled training fits
train images without generalizing held-out zero/nonzero calibration. This is a
supervision-design failure, not a failure of the EF-LIC insertion point.

Updated next experiment:

1. Generate slice/spatial teacher labels from E236 local-policy artifacts or
   local residual/quantization-error deltas, rather than broadcasting one image
   label across every map location.
2. Split the controller into a binary activation head and a conditional
   family/strength head. Keep zero/fallback as the default state.
3. Add richer decoder-safe local maps if needed: calibrated scale/mean maps,
   support-prefix summaries, z-prior summaries, and local uncertainty maps.
4. Evaluate held-out false-positive nonzero rate before any codec-loop or full
   fine-tuning claim.
5. Only after E242 controls activation should the learned head be inserted into
   EF-LIC compress/decompress for full Kodak/CLIC/Tecnick evaluation.

This keeps the conference path conservative: E241 prevents us from promoting a
plausible-looking but unsafe classifier, while E242 targets the actual
HCG-RVQ claim that quantizer geometry should be locally generated and
confidence-gated by the hyperprior/context state.

## EF-LIC E242 Spatial Teacher Result and Two-Stage Controller Plan

E242 should become the new supervision basis for EF-LIC HCG integration. It
exports exact E236 policy alpha maps alongside decoder-safe E240 context maps,
producing spatial `target_map` labels instead of broadcasting image labels.
This is closer to the intended method and preserves zero/fallback positions.

Do not continue with a single multiclass head as the main controller. The E242
frozen-head audit shows it flips between zero-collapse and all-active collapse
when the activation loss balance changes. The next code unit should implement:

1. `LocalHCGActivationHead`: binary zero-vs-active logits from the same context
   maps, initialized toward zero and trained with false-positive-heavy loss.
2. Threshold calibration on a held-out split: report active recall, zero
   false-positive rate, and PR/AUROC before codec-loop evaluation.
3. `LocalHCGFamilyStrengthHead`: conditional family and alpha/strength prediction
   trained only on active teacher positions.
4. EF-LIC compress/decompress insertion only after the activation head controls
   held-out false positives.
5. Full Kodak/CLIC/Tecnick RD and perceptual evaluation after the two-stage head
   passes the frozen-head calibration gate.

This preserves the high-level claim while being honest about the current
experimental evidence: local geometry supervision is now available, but reliable
activation calibration is the main remaining blocker before paper-grade codec
results.

## EF-LIC E243 Result and Next Controller Upgrade

E243 implements the first binary activation head, but it should not yet be put
into EF-LIC compress/decompress as the paper-main learned controller. The
four-fold Kodak24 audit shows that the current local/global-summary features do
not generalize activation calibration strongly enough.

Updated implementation policy:

1. Keep the two-stage design. Activation calibration remains the gate before
   family/strength prediction.
2. Improve activation inputs before full codec-loop training: add a true global
   pooling branch, z-prior or q-index summaries, slice-state summaries, and
   possibly CLIC/OpenImages fit labels instead of Kodak-only folds.
3. Keep EF-LIC's final R-D/perceptual objective clean. Auxiliary activation
   losses are for frozen-head pretraining or warmup; final claims must include
   ablations showing the HCG branch, not auxiliary loss changes, causes the gain.
4. For GLC, reuse the same lesson: sparse active branch plus scalar fallback is
   promising, but the next branch must be reliability-controlled and bit-aware
   before Stage-II/III-compatible fine-tuning.

The next concrete EF-LIC code unit should be an activation head with a global
pooling path and richer decoder-safe context summaries, followed by the same
four-fold threshold audit. Only if held-out FPR/recall improves should the
family/strength head and codec-loop RD evaluation resume.

## EF-LIC Controller Strategy After E244/E245

Do not insert the E244 cross-slice activation head into EF-LIC compress/decompress
as a paper-main controller. It overfits Kodak24 teacher maps and does not
transfer thresholds reliably. Also do not keep increasing activation-head
capacity on the same Kodak-only labels; E245 shows that the available local
mean/scale/support channels themselves have weak conservative separation.

The next EF-LIC strengthening path should be one of these, in priority order:

1. Generate independent activation/family/strength labels on a non-Kodak fit
   split, then reserve Kodak/CLIC/Tecnick for evaluation.
2. Add richer decoder-safe signals to the controller: z-prior summaries,
   predicted entropy/index-rate summaries, slice-sequential state, and local
   quantization-error proxies that are available or reproducible at decoding.
3. Keep the final codec objective R-D/perceptual dominated. Use activation and
   family losses as weak pretraining or auxiliary calibration, not as heavy
   losses that can overpower the original compression objective.
4. Only after conservative activation improves should the conditional
   family/strength head and full-training/full-evaluation runs be promoted to
   paper-main evidence.

For GLC, the same rule applies. Sparse active-state experiments are useful, but
the next strong GLC variant should include bit-aware/index-aware reliability
control rather than a larger active branch alone.

## Full-Training Gate After CLIC E244/E245

The CLIC professional tensors are now available and should be used as an
independent fit/calibration source, but they do not by themselves justify full
training of the current activation head. Promotion criteria before full EF-LIC
training are:

1. Add decoder-known signals beyond local mean/scale/support, especially
   z-prior/index-rate summaries or branch-cost proxies.
2. Show image-held-out or dataset-held-out activation calibration that improves
   over E243/E244 without all-active FPR collapse.
3. Keep the full-training loss close to the original EF-LIC R-D/perceptual loss;
   activation/family terms should initialize or weakly regularize, not dominate.
4. Evaluate full checkpoints on Kodak, CLIC professional, CLIC mobile if used,
   and Tecnick/DIV2K-style external splits with unchanged or explicitly counted
   side information.

## Full-Training Gate After E246

E246 tested the direct "add z/index and richer decoder-safe summaries" route.
That route should not be promoted as-is. The features have strong in-table
capacity but poor held-out activation and family transfer, so the next
implementation should avoid a heavy frozen-teacher controller loss.

Updated priority:

1. Keep original EF-LIC R-D/perceptual training dominant for any codec-loop
   experiment. HCG activation/family losses may initialize or weakly regularize
   the branch, but they should not define the final objective.
2. Preserve a conservative fallback path: scalar/zero baseline, all-on HCG,
   fixed guarded HCG, teacher/oracle upper bound, and learned HCG must all be
   comparable under the same evaluation protocol.
3. If using supervised controllers, generate larger independent fit labels
   outside the final test sets. E242/E246 labels are useful diagnostics, but
   the small Kodak/CLIC teacher table is not sufficient as a paper-main
   controller dataset.
4. For EF-LIC, implement the next branch as a decoder-safe, state-preserving
   local geometry module whose effect is judged by RD/perceptual metrics,
   codebook/residual statistics, and activation calibration.
5. For GLC, prioritize bit-aware/index-aware reliability and active-branch
   accounting before scaling to full training. Residual-MSE gains alone are not
   enough if empirical index entropy or perceptual metrics regress.

This keeps the two-track plan intact: controlled evidence continues to map the
mechanism and failure modes, while full-training candidates are selected only
when they can test HCG-RVQ's core claim under a codec objective rather than a
small teacher-label shortcut.

## Loss Policy After E247

Use the E247 audit as a guardrail when selecting EF-LIC/GLC full-training
configs:

1. Paper-main rows should start from RD/commit-only or explicitly weak-auxiliary
   configs.
2. Teacher/selector/anchor-heavy configs are diagnostic or warmup rows unless a
   matched ablation proves the auxiliary loss is not carrying the gain.
3. Report auxiliary loss weights in every full-training table and keep a clean
   baseline with the same backbone, bitrate target, checkpoint policy, and
   evaluation split.
4. For large-risk designs, run a short smoke first, then a mid-scale training
   gate, then full training only if RD/perceptual metrics and intermediate
   statistics move in the same direction.

This addresses the full-training concern: short-cycle experiments are useful
for choosing mechanisms, but the final evidence must be recreated under a
codec-objective-dominant training recipe.

## Implementation Gate After E248

E248 promotes two concrete next tracks and rejects two tempting shortcuts.

Promoted EF-LIC track: implement a decoder-safe compact/local HCG branch with a
zero/scalar fallback. The branch should be trained in the codec loop with the
original EF-LIC R-D/perceptual objective dominant. Teacher labels may initialize
or weakly regularize activation/family decisions, but the final claim must come
from matched codec evaluation. The first promoted run should compare baseline,
zero/scalar fallback, all-on HCG, fixed guarded HCG, oracle/teacher upper bound,
and learned HCG under the same split and checkpoint policy.

Rejected EF-LIC shortcut: do not promote the frozen E244/E246 teacher/selector
classifier as paper-main. It is useful for diagnostics and warmup, but E246
shows that current feature groups do not generalize reliably enough.

Promoted GLC track: implement a bit-aware q0 tail VQ/HCG branch over active
residual states. The branch must report empirical index bpp, PSNR, MS-SSIM,
LPIPS, DISTS, active residual MSE, codebook usage, and dead-code statistics. Use
q0 first, then expand to q1-q3 or multi-stage RVQ only after bpp/perceptual
conflicts are controlled.

Rejected GLC shortcut: dense/all-on or residual-MSE-only RVQ remains a mechanism
probe. It should not be used as the paper-main variant because E181 shows that
PSNR/MS-SSIM/LPIPS improvements can coexist with DISTS and bpp regressions.

Practical rule: short-cycle experiments remain useful for selecting mechanisms,
but every paper-main row must be regenerated under a clean, codec-objective-dominant recipe and a full evaluation protocol. CUDA jobs on this machine must
remain pinned to GPU0.

## GLC Bit-Aware Gate After E249

Do not scale the E181 branch directly into full training. Its trained q0 branch
passes a tolerant perceptual gate but fails the strict bpp/DISTS gate. The next
GLC implementation should add one or more of the following before promotion:

1. A branch objective or selection rule that penalizes empirical index bpp.
2. A reliability gate that activates q0 RVQ only where `DISTS + 3*LPIPS` gain is
   large enough to pay for the index rate.
3. Smaller or shared codebooks for low-gain active groups, with K=4 as a
   rate-lighter ablation and K=8 as a headroom ablation.
4. A full evaluation table that separates PSNR/MS-SSIM/LPIPS gains from DISTS
   and bpp costs.

The target remains simple: local HCG/RVQ should improve the codec objective, not
win by adding a large auxiliary loss. Use E249's break-even bpp weight as a
promotion metric for future GLC branches.

## GLC Implementation Gate After E250

E250 changes the GLC next step from "add a soft entropy penalty and scale" to
"scale the K=8 part-group LPIPS+DISTS+soft-index branch." Both the best OI8 ->
Kodak8 row (`score1=-0.006642`) and the matched OI16 -> Kodak8 row
(`score1=-0.003769`) beat the E249/E181 bpp-charged promotion score
(`-0.002429`). The matched row also reduces empirical bpp delta
(`+0.014432` versus `+0.014548`) and has no nonfinite rows.

Do not promote K=4 or shared-codebook branches as the main GLC route. They lower
rate, but the quality loss shows that they remove the local residual geometry
that HCG-RVQ is meant to exploit. Keep them as rate lower-bound ablations.

Next GLC implementation target:

1. Keep q0, K=8, part-group active residual codebooks as the headroom branch.
2. Include LPIPS/DISTS plus the original codec distortion term in the local gate;
   do not add large unrelated auxiliary losses.
3. Keep the memory-safe runner behavior: free eval tensors before training and
   backpropagate per image so OI16/Kodak8 and larger gates fit on GPU0.
4. Use full Kodak to harden the selective activation/index-prior gate; do
   not scale the all-on branch directly to CLIC Professional.
5. Add explicit activation/index-prior control before any CLIC-scale run, since
   DISTS remains the limiting metric on full Kodak/CLIC.
6. Continue reporting PSNR, MS-SSIM, LPIPS, DISTS, empirical bpp, active residual
   MSE, codebook entropy, used/dead code fraction, and per-image wins.

This keeps the HCG-RVQ claim aligned with the prompt: hyperprior-conditioned
local quantizer geometry is promising only if the model learns where the extra
geometry is worth its bitrate.

## GLC Selective Activation Gate After E251

E251 changes the immediate GLC implementation target again: keep the E250 q0
K=8 part-group branch, but do not scale it as an all-on module. Full Kodak24
all-on is mixed (`+0.002014` score), while oracle activation is clearly useful
(`-0.005773`) and leave-one-out single-feature gating remains negative
(`-0.001565`).

Next implementation target:

1. Add a selective activation/index-prior head for the q0 E250 branch.
2. Use codec-side or signalable features first: active residual magnitude,
   local hyperprior scale, predicted index entropy, codebook uncertainty, and
   optional low-cost side bits. The E251 safe-feature threshold is only
   in-sample useful and fails leave-one-out, so this should be a learned or
   hyperprior-predicted gate rather than a hand threshold.
3. Keep base PSNR/LPIPS/DISTS features as analysis-only probes, not as deployable
   controller inputs.
4. Train with the original GLC objective plus only minimal VQ/index terms; avoid
   adding large teacher or geometry losses that could become the real source of
   improvement.
5. Evaluate policies as baseline, all-on, oracle, simple threshold, and learned
   gate on Kodak24 before CLIC Professional scaling.

This route preserves the strong part of E250, namely local K=8 residual geometry
capacity, while directly addressing the two remaining failure modes: unnecessary
index rate on low-gain images and DISTS regressions on hard images.

## GLC External-Gate Strategy After E252/E253

Do not run the current E250 branch as a larger all-on CLIC Professional or full
training candidate. E252/E253 show that the Kodak-positive branch fails the
CLIC Professional first-8 external gate even under oracle activation (`0/8`
selected). The next GLC implementation should therefore be promoted only after a
reliability mechanism is trained or calibrated on a domain-mixed split.

Required next GLC branch changes before another CLIC-scale run:

1. Add a small hyperprior/index-prior activation head that predicts branch use
   before paying index bits.
2. Train or calibrate it with a split that includes OpenImages and CLIC-style
   images; keep Kodak as one reporting set rather than the only design target.
3. Use DISTS as a hard validation guard, not only LPIPS or PSNR.
4. Report the same policies on each domain: baseline, all-on, oracle, safe
   threshold, learned gate, and no-branch fallback.
5. Keep E252/E253 as negative controls showing why all-on q0 tail VQ is not the
   submission method.

This keeps the path ambitious without hand-waving away the hard domain result:
HCG-RVQ is promising when it creates an alternate local coding mode, but the
paper-grade method must also learn when that mode should be silent.

## GLC Domain-Mixed Controller Strategy After E254

E254 blocks the naive next step. Do not promote the E250 q0 K=8 part-group
branch as all-on full training, and do not trust a single hand threshold learned
on Kodak. The pooled oracle still proves useful local-RVQ headroom, but
leave-domain-out shows the current simple gate is not robust across Kodak and
CLIC Professional.

Next GLC implementation target:

1. Build a small reliability/index-prior controller for the E250-style local RVQ
   branch, using branch-internal signals such as active residual magnitude,
   predicted index entropy, codebook uncertainty, local scale, and optionally a
   low-cost signaled fallback bit.
2. Train/calibrate the controller on a domain-mixed split that includes
   OpenImages-style training images and CLIC-like validation images; use Kodak as
   a reporting set, not the only design target.
3. Keep no-branch fallback as a first-class action. The controller must be
   allowed to silence the branch when DISTS or bpp risk is high.
4. Keep the loss simple: original codec R-D/perceptual objective, VQ commitment
   or index entropy terms only where they are directly needed, and weak warmup
   terms if stability requires them.
5. Before any expensive full-training run, require: negative score on Kodak24,
   non-harmful CLIC Professional validation, no nonfinite rows on GPU0, and a
   per-image analysis showing which images are selected and why.

This keeps the HCG-RVQ route ambitious but controlled: the next method should be
not more VQ everywhere, but a hyperprior-conditioned local coding mode with a
simple, explainable reliability controller.

## GLC Controller Proxy Strategy After E255

E255 confirms the next implementation should not be a larger offline classifier.
A tiny linear/score controller can recover weak leave-one-image-out headroom, but
cross-domain transfer either becomes silent or harmful. Therefore the next useful
code path is to create better controller evidence, not to increase controller
capacity on the same E254 rows.

Next GLC actions:

1. Generate CLIC-like calibration labels for the E250-style q0 K=8 branch, using
   images not reserved for final reporting.
2. Train a small reliability/index-prior head with no-branch fallback and
   DISTS+bpp risk as the threshold-selection objective.
3. Keep `loocv_branch_plus_rate_score_regressor` as a reporting baseline and
   warmup target, not as a final method.
4. Promote to codec-loop/full-training only if the learned controller is negative
   on Kodak24 and non-harmful on CLIC Professional validation under the same
   score.

## GLC Controller Strategy After E256-E258

E256-E258 keep the GLC route alive but block another all-on or offline-controller
promotion. A non-overlapping CLIC Professional slice gives one oracle-positive
image, yet all-on is highly harmful and leave-domain-out controller transfer is
still not reliable.

Next GLC actions:

1. Treat CLIC calibration as necessary training evidence, not as a final reporting
   set. Keep separate CLIC Professional rows reserved for external evaluation.
2. Move from offline thresholds to a codec-adjacent reliability/index-prior head
   whose objective is DISTS+bpp risk with no-branch fallback.
3. Keep the actual loss simple: original GLC reconstruction/perceptual objective
   plus minimal VQ commitment/index entropy terms. Do not add large teacher or
   selector losses to the final full-training claim.
4. Promotion gate before full training: negative Kodak24 score, non-harmful CLIC
   Professional score, nonzero but controlled activation, and no nonfinite rows
   on GPU0.
5. If more calibration labels are needed, use additional CLIC/OpenImages-like
   slices and report which slices are calibration versus final evaluation.

## Full-Training Promotion Rule After E259

E259 turns the current short-cycle evidence into a concrete promotion rule.
Dense/all-on HCG-RVQ insertion is not the next full-training target for either
EF-LIC or GLC. Both tracks should instead move toward the same compact method:

1. A local HCG-RVQ branch whose shift/scale/geometry/index information is
   conditioned by hyperprior or codec context.
2. A small reliability/index-prior controller with no-branch fallback.
3. The original EF-LIC/GLC codec objective as the dominant loss, with only direct
   VQ commitment and index-entropy terms added as needed.
4. Promotion only after a mid-scale codec-loop run is finite on GPU0, avoids
   dead-code collapse, improves or at least does not harm the guarded
   DISTS+bpp/LPIPS score on CLIC Professional, and remains negative on Kodak24.

This preserves the ambitious direction: do not simply add more VQ everywhere.
Make the quantizer geometry conditional, rate-aware, and allowed to stay silent
when the branch cannot pay for its own index/perceptual cost.

## Controller Implementation Contract After E260

Use `hcg_rvq/reliability_index_controller.py` as the shared controller contract
for the next EF-LIC and GLC integrations.

Required behavior:

1. The controller outputs an activation logit and a signed risk/score estimate.
2. Initial bias must prefer zero/fallback.
3. Selection must go through an explicit no-branch fallback mask.
4. False-positive activation must be more expensive than missed activation.
5. Risk/score prediction is optional during warmup, but any paper-main GLC/EF-LIC
   version must account for index/rate cost before activating the HCG branch.

E260 shows why this should be trained in the codec loop rather than only as an
offline classifier. The same compact MLP reaches oracle in resubstitution but is
harmful under LOOCV and leave-domain splits. Therefore, the next implementation
step should insert this controller contract into a mid-scale EF-LIC/GLC branch
with the original codec loss dominant and domain-mixed calibration/evaluation.

## Domain-Robust Calibration Rule After E261

E261 tightens the promotion rule. A hand threshold over current GLC branch
diagnostics is not robust enough for paper-main use: it is useful only in
resubstitution and turns harmful under LOOCV, leave-domain, and leave-variant
checks. Therefore the next implementation should not be another threshold sweep
or an all-on full-training run.

Updated GLC/EF-LIC integration target:

1. Insert the E260 reliability/index controller contract into the codec-loop
   branch rather than fitting it after evaluation rows are collected.
2. Preserve no-branch fallback as the default initialization and as a valid
   action during training/evaluation.
3. Keep the loss direct: original codec R-D/perceptual objective first, then VQ
   commitment/index entropy/risk terms only when they directly support rate-aware
   activation.
4. Treat E261 threshold policies as ablation/negative controls in the paper.
5. Promote to full training only when a mid-scale run shows non-harmful CLIC
   behavior, negative Kodak behavior, finite GPU0 execution, stable codebook
   usage, and interpretable selected/blocked image cases.

This is still an ambitious route. The lesson is not to abandon HCG-RVQ; it is to
make the hyperprior/context control both the quantizer geometry and the decision
to spend extra index bits.

## Fallback-Mix Implementation Contract After E262

Use `mix_with_fallback` from `hcg_rvq/reliability_index_controller.py` as the
shared EF-LIC/GLC branch insertion primitive.

Required usage for the next pilots:

1. Compute the original codec output and the HCG-RVQ branch output with matched
   shapes.
2. Predict `active_logit` and optional `risk_score` from decoder-safe or
   branch-local context.
3. During training, use soft `mix_with_fallback(..., hard=False)` so the branch
   and controller receive gradients.
4. During deterministic evaluation/bit accounting, use hard fallback and report
   selected fraction, selected image cases, and no-branch exactness.
5. Keep all-on and threshold-only policies as ablations, not as the main method.

This is the immediate implementation path toward full training: conservative
fallback first, then learned local geometry/index activation, then mid-scale
Kodak/CLIC validation before expensive full runs.

## Full-Training Promotion Rule After E263

E263 promotes one design to the top of the full-training queue: GLC/EF-LIC
branches should use codec-loop-trained fallback gating, not dense all-on RVQ and
not post-hoc offline thresholds.

Promotion criteria before any expensive full run:

1. Reproduce the E263 pattern on a larger Kodak subset and a CLIC Professional
   subset: all-on remains an ablation, soft fallback improves the guarded score,
   and hard fallback exactly recovers the base when inactive.
2. Replace diagnostic soft bpp with final model-appropriate bit accounting or a
   conservative upper-bound bit estimate.
3. Calibrate a hard/annealed evaluation policy so selected cases are non-harmful
   under held-out images, not only under soft blending.
4. Port the same controller contract to EF-LIC with spatial context features,
   preserving the original EF-LIC RD/perceptual objective as the dominant loss.
5. Keep the loss small: original codec objective, VQ commitment/usage only where
   needed, and direct rate-aware gate penalties. Do not add broad teacher or
   auxiliary losses unless an ablation proves they improve the final codec
   metric.

E263 already satisfies the first promotion criterion on Kodak4 and CLIC
Professional first-8 as a short-cycle smoke: soft fallback is beneficial on all
evaluated images, dense all-on remains harmful, and no nonfinite rows appear on
GPU0. The next implementation step should therefore be either a larger
held-out-slice GLC run with final bit-accounting work, or the EF-LIC spatial
port of the same fallback-gated controller contract.

This keeps the method simple and paper-explainable while still aiming for a
large improvement: conditional local quantizer geometry should be useful because
the model learns when it is worth spending extra residual/index capacity.

## Rate-Accounting Promotion Rule After E264

E264 makes the promotion rule stricter. Diagnostic gate-scaled bpp is useful for
design search, but it is not enough for a paper benchmark. The next GLC/EF-LIC
implementation promoted toward full training must satisfy one of these
accounting contracts:

1. Hard or annealed sparse activation reports actual selected-index bpp and
   proves selected cases are non-harmful on held-out Kodak and CLIC
   Professional slices.
2. A progressive or entropy-coded branch transmits only the activated residual
   information, so the measured branch bpp tracks the learned gate instead of
   silently paying full all-on cost.

Until one of these is implemented, E263/E264 should be framed as strong design
evidence rather than final RD-curve evidence. This is still progress: the
geometry branch is useful, all-on is the wrong control policy, and the next
engineering target is now sharply defined.

The EF-LIC port should reuse the same rule. Its spatial controller can be
smoked with decoder-safe context maps immediately, but final promotion requires
real selected-index/rate reporting, not only a soft tensor blend.

## EF-LIC Port Rule After E265

E265 passes the EF-LIC controller smoke, so the next EF-LIC work should move
from teacher-artifact wiring to real codec-loop insertion.

Required next unit:

1. Use decoder-safe context maps after EF-LIC mean/scale prediction to drive
   `SpatialReliabilityIndexHead`.
2. Build a small local HCG/RVQ branch whose output shape exactly matches the
   original EF-LIC slice or reconstruction residual being modified.
3. Blend through `mix_with_fallback`, with hard fallback proving exact base
   recovery.
4. Report all-on, soft gate, hard gate, and base under the same Kodak/CLIC
   images.
5. Include real or conservative selected-index/rate accounting from the start;
   do not rely on soft-gate tensor scaling as final bpp.

This is the EF-LIC analogue of the E263/E264 GLC lesson: the local quantizer
branch is allowed to be useful, but the paper-main method must learn when it is
worth paying for it.

## Low-Rate GLC Promotion Rule After E267

E267 updates the full-training candidate. The top GLC branch is no longer the
original E263 dense branch; it is the low-rate fallback-gated branch with
`K=4` and active parts `[0, 1]`.

Promotion rule:

1. Keep the original codec path as the default fallback and keep all-on as a
   negative-control ablation.
2. Train the compact branch with the original codec objective dominant; add only
   minimal VQ/index/rate terms that directly support reliable activation.
3. Replace diagnostic gate-scaled bpp with selected-index bpp or a progressive
   branch whose measured rate scales with the activated residual information.
4. Re-run Kodak and CLIC Professional split tests with checkpoint evaluation,
   codebook usage, residual-stage contribution, and failure-case analysis.
5. Promote to expensive full training only if the low-rate branch remains
   negative under selected-index/progressive accounting on held-out CLIC as well
   as Kodak.

This also changes the EF-LIC port target. EF-LIC should not start with a dense
all-on HCG/RVQ branch. It should mirror the low-rate GLC design: small local
branch, decoder-safe spatial reliability/index head, fallback exactness, and
real selected-index/rate reporting from the first codec-loop pilot.

## E268 Promotion Update

E268 promotes the low-rate GLC branch from a promising held-out probe to the
current primary GLC implementation target.

Immediate next actions:

1. Implement actual selected-index or progressive branch rate accounting for
   the `K=4`, parts `[0, 1]` branch.
2. Add checkpoint-level reporting for the low-rate branch: base, all-on,
   soft gate, hard/annealed gate, selected-index bpp, codebook usage,
   residual-stage contribution, and failure cases.
3. Run a larger but still bounded Kodak/CLIC Professional pilot before any
   full-training launch.
4. Port the same low-rate branch shape to EF-LIC, using the E265 decoder-safe
   context controller rather than a dense all-on branch.
5. Keep the loss compact: original codec objective dominant, plus only the VQ
   commitment/index/rate terms needed to make selected activation real.

Full training becomes justified after the selected-index/progressive audit is
negative on held-out CLIC as well as Kodak. Until then, E268 is strong design
evidence and the best current HCG-RVQ strengthening direction, not a final
RD-curve result.

## E269 Selected-Rate Implementation Target

E269 narrows the next implementation target to a concrete contract:
low-rate fallback-gated GLC (`K=4`, active parts `[0, 1]`) should become an
actual selected-index or progressive-rate codec path.

Required properties for the next GLC step:

1. Preserve exact fallback to the base reconstruction when the branch is not
   selected.
2. Transmit branch indices only for selected/progressive residual information,
   then report measured selected-index bpp rather than diagnostic gate-scaled
   bpp.
3. Keep the original codec objective dominant. Add only minimal VQ commitment,
   index/rate, or activation terms needed to make the selected branch real.
4. Report base, all-on negative control, soft gate, hard/annealed gate, and
   selected/progressive rate on the same Kodak and CLIC Professional slices.
5. Include codebook usage, dead-code fraction, index entropy, residual-stage
   MSE contribution, and the hardest CLIC rows identified by E269.

The first success criterion is no longer vague: the selected/progressive version
should keep the trained low-rate full-accounting score near the E269 budget
(`-0.011600` before replacing the accounting proxy) and remove or fallback the
`casey-fyfe-999.png` tail where full-rate accounting is `+0.000406`. Only after
that should this branch move to larger/full training.

EF-LIC should mirror this design rather than testing a dense all-on branch. The
E265 controller contract is ready; the EF-LIC branch should start small,
fallback-safe, and rate-accounted from the first codec-loop pilot.

## E270 Rate-Cap First Implementation Rule

E270 updates the next selected-rate implementation rule: start simple. The first
real selected/progressive GLC branch should use an encoder-side branch-rate cap
before adding a learned multi-feature selector.

Concrete next implementation target:

1. Compute the local low-rate branch candidate as in E267/E268 (`K=4`, active
   parts `[0, 1]`).
2. Estimate or measure the branch index cost for the candidate.
3. Select the branch only when the estimated branch dbpp is below a calibrated
   cap; otherwise emit exact fallback.
4. Report selected-index bpp, selected fraction, selected win rate, and failure
   rows under first-to-held and Kodak-to-CLIC transfer.
5. Keep all-on as a negative control and keep the original codec objective
   dominant; do not add extra reliability losses unless the simple rate cap
   fails on larger splits.

The starting cap values from E270 are `0.00303411` for first-to-held and
`0.00242854` for Kodak-to-CLIC. These are not final constants; they are initial
paper-prototype calibration points for the next codec-loop implementation.

## E271 Progressive-Soft Requirement

E271 adds one more constraint to the selected-rate implementation: the rate cap
must be paired with soft/progressive residual amplitude control. The cap by
itself does not rescue all-on output; capped all-on remains harmful on both CLIC
and Kodak, while capped soft output remains useful.

Next implementation contract:

1. Keep the low-rate branch small (`K=4`, active parts `[0, 1]`) unless a new
   branch first passes the same rate-margin audit.
2. Encode selected/progressive branch information so that the measured bpp
   follows the amount of residual information actually used.
3. Preserve exact fallback when the branch is not selected.
4. Keep all-on output as a negative-control ablation, not as the promoted
   method.
5. Avoid adding broad auxiliary losses. The main objective should remain the
   codec's original RD/perceptual objective plus minimal VQ/index/rate terms
   needed to make the selected/progressive branch trainable and measurable.

The immediate GLC task is to replace the soft tensor-blend proxy with an
actual selected/progressive index path and then re-run the E269/E270/E271
reports with measured selected-index bpp, codebook usage, residual-stage
contribution, and failure-row analysis. The immediate EF-LIC task is to port
this same low-rate, fallback-safe, progressive-soft contract rather than trying
a dense all-on branch first.

## E272 Gate-Overhead Constraint

E272 constrains how the progressive-soft branch should be made real. Scalar or
coarse selection signaling is acceptable in the current audit; dense gate maps
are risky, especially on CLIC.

Implementation rule for the next selected/progressive path:

1. First try decoder-predicted reliability from hyperprior/context plus an
   encoder-side rate cap.
2. If side information is needed, start with scalar or coarse tile signaling,
   and charge those bits explicitly.
3. Avoid a dense gate map as the main design unless it shows a clear gain under
   measured side-rate accounting.
4. Report gate-side bpp separately from branch-index bpp so the paper can show
   whether the gain comes from quantizer geometry or hidden control overhead.
5. Reuse the same accounting on EF-LIC before any full-training claim.

## E273/E274 Bitstream Promotion Gate

E273/E274 tighten the selected/progressive implementation contract. The current
main GLC branch remains `K=4`, active parts `[0, 1]`, but it should not be
promoted as a naive base-plus-full-RVQ enhancement. That conservative accounting
is too expensive on CLIC.

Promoted next implementation paths:

1. Selected replacement branch: when active, avoid transmitting both the scalar
   active residual stream and the RVQ active residual stream. Report
   `active_scalar_bpp`, `active_rvq_extra_bpp`, and replacement delta bpp.
2. Fractional or stage-wise progressive branch: transmit only the subset of
   active RVQ information that survives the E274 margin. The first CLIC target is
   about half of the measured active RVQ extra bpp, while Kodak can tolerate more.
3. Entropy/index-prior branch: reduce active RVQ index cost using local
   part/group priors before increasing codebook capacity.

Blocked or diagnostic-only paths:

1. Dense all-on output remains a negative control.
2. Dense local gate maps are not mainline unless their side bpp is explicitly
   measured and still improves CLIC.
3. Base-plus-full-active-RVQ enhancement is a conservative audit row, not the
   current paper-main method.

Promotion criteria before full training:

1. Negative guarded score on both CLIC Professional held-out and Kodak under
   measured selected/replacement/progressive bpp.
2. No nonfinite rows on `CUDA_VISIBLE_DEVICES=0` / `cuda:0`.
3. Explicit accounting of active scalar bits, active RVQ bits, gate side bits,
   codebook usage/perplexity, dead-code fraction, and residual-stage gains.
4. Same split protocol for base, all-on, soft/progressive, selected/replacement,
   and oracle/failure analyses.
5. Original GLC/EF-LIC objective remains dominant; auxiliary losses are limited
   to direct VQ commitment/index/rate support or short warmup.

This is the bridge from short-cycle evidence to paper-grade experiments: the
method is not simply more VQ, but hyperprior/context-conditioned local quantizer
geometry that is allowed to stay silent or replace scalar residual coding only
when its measured index cost is justified.

## E275 Selected-Replacement Implementation Target

E275 promotes selected replacement as the immediate GLC engineering target.
Base-plus-full-active-RVQ enhancement is too expensive on CLIC, but replacing
active scalar residual bits with active RVQ bits gives negative mean scores on
both CLIC and Kodak in the current audit.

Concrete next code unit:

1. Add a selected replacement mode to the GLC pilot path: active locations use
   the low-rate RVQ branch, inactive locations keep the original scalar path,
   and the reported bpp charges active RVQ indices instead of active scalar
   residual bits for the selected subset.
2. Keep a replacement dbpp cap around `0.0025` as the first conservative CLIC
   guard, then calibrate it on a fit split and transfer to held-out CLIC/Kodak.
3. Preserve exact fallback for unselected images or regions.
4. Report `casey-fyfe-999.png` and any similar rows as failure cases, with
   quality margin, active scalar bpp, active RVQ bpp, replacement dbpp, gate
   mean, and codebook entropy.
5. Only after selected replacement remains negative under measured bit accounting
   should this branch move to longer/full training.

EF-LIC should mirror this lesson: avoid all-on or extra-stream framing first;
start with a state-preserving local replacement branch and explicit fallback.

## E276 Direct Replacement-Row Rule

`tools/run_e263_glc_fallback_gate_codec_loop_pilot.py` now emits replacement
rows directly. Future GLC pilots should always include these rows when testing
low-rate active branches:

- `replacement_soft`
- `replacement_all_on`
- `rate_cap_replacement_soft`
- `rate_cap_replacement_all_on`

Default first cap: `--replacement-cap-dbpp 0.0025` for CLIC-facing short-cycle
runs, with transfer/calibration required before paper claims.

Promotion rule update: replacement rows should be the primary short-cycle metric
for the GLC low-rate branch. Progressive-extra rows remain a conservative
stress-test, and all-on rows remain negative controls. Full training should not
start until direct replacement rows remain negative on a larger held-out CLIC
slice with codebook usage and residual-stage diagnostics.

## Update After E277/E278: Promote Selected Replacement, Not Additive Enhancement

The GLC integration path should now prioritize selected active replacement.
Across E276/E277, replacement accounting stays negative on CLIC and Kodak while
additive progressive accounting is weak or harmful. The next implementable
candidate is:

- Base GLC scalar path remains exact for inactive regions.
- Active regions use the HCG-RVQ local residual branch as a replacement mode,
  not as a second stream added on top of scalar residual bits.
- The encoder selects the replacement mode only when the measured replacement
  delta-bpp is below a calibrated cap. The current safest next cap is around
  `0.0035`, not the earlier overly conservative `0.0025`.
- The decoder obtains the same selection through hyperprior/context prediction
  or a coarse signaled fallback bit, not a dense gate map.
- The loss should stay simple: original codec RD/perceptual terms plus direct
  VQ/index/rate terms needed to make replacement measurable. Do not add broad
  auxiliary losses unless a specific diagnostic shows a failure mode they solve.

Before any paper-main full-training claim, run the promoted candidate with
paper-aligned training/evaluation and log checkpoint-wise RD, CLIC Professional
and Kodak curves, active scalar/RVQ/replacement bpp, codebook usage, residual
stage contribution, and failure cases. Keep all-on and additive-progressive rows
as ablations because they make the design choice explainable.



## Update After E279/E280: Safe and Aggressive Replacement Caps

The GLC pilot now supports multiple replacement caps in one run via
`--replacement-cap-dbpp-values`. Future selected-replacement experiments should
emit at least three controller rows in the same table:

- legacy conservative cap `0.0025`, mostly for continuity with E276/E277;
- safe paper-facing cap `0.0035`, currently selected-win `1.000000` over the
  53-image E280 aggregate;
- aggressive cap `0.0040`, currently the best mean but not tail-safe on CLIC.

This lets the implementation keep a simple controller while documenting the
tradeoff between conservative reliability and larger performance gains. The
next full-training candidate should not choose only one cap during design
search. It should log both `0.0035` and `0.0040`, keep all-on and additive-extra
negative controls, and then decide the paper-main cap from held-out CLIC/Kodak
curves plus worst-case images.

Concrete next step: make replacement accounting closer to the eventual bitstream
by avoiding double transmission of active scalar residual and active RVQ indices
in selected regions. Once that path is implemented, repeat the E280 table at
checkpoint scale and then decide whether the GLC candidate is ready for
paper-aligned full training.


## Update After E281: Empirical vs Fixed-Index Replacement Accounting

E281 adds a stricter promotion rule for GLC and EF-LIC. A replacement-mode branch is not paper-ready merely because it is negative under empirical index entropy. It must also report a conservative fixed-index bound, or explicitly implement the selected-index/coarse-index coding that justifies the empirical rate.

Current GLC status:

- Empirical replacement remains strong over CLIC33 + Kodak20: score `-0.009700`, win fraction `0.924528`.
- Fixed-index replacement remains negative but weaker: score `-0.007449`, win fraction `0.773585`.
- CLIC is the bottleneck: empirical score `-0.007603`, fixed-index score `-0.004969`, fixed-win fraction `0.636364`.
- Kodak is robust under both accounting modes: empirical `-0.013159`, fixed `-0.011542`, fixed-win `1.000000`.

Implementation rule:

1. Treat `0.0035` as the conservative selected-replacement cap for paper-facing controlled evidence.
2. Carry `0.0040` as an aggressive branch, but do not make it the only paper-main policy unless the selected-index path improves CLIC tail reliability.
3. In every GLC/EF-LIC pilot, report empirical replacement score, fixed-index replacement score, selected fraction, selected win fraction, active scalar bpp, active RVQ bpp, fixed-index penalty, and worst cases.
4. For EF-LIC specifically, preserve the original entropy-coding-free claim: if HCG-RVQ uses any non-fixed index statistics or coarse signaling, account for it explicitly. Otherwise the fixed-index bound is the main rate claim.

Next concrete code target: implement an explicit selected replacement path rather than only a reporting row. Selected regions should not transmit both scalar residual and RVQ indices. The decoder should recover selection from decoder-safe hyper/context maps or from a coarse signaled state with measured overhead. The original codec objective remains dominant; new losses should be limited to the VQ/index/rate terms needed to make this replacement mode measurable.

## E282 Controller Transfer Update

GLC replacement should now be implemented with two explicit controller modes:

1. Conservative paper mode: cap around `0.0035` when empirical/coarse index
   accounting is allowed, with fixed-index ablation reported.
2. Strict fixed-index mode: cap around `0.0030` for EF-LIC-style no-entropy
   claims or for a fully fixed-length GLC ablation.

Do not tune the controller only on Kodak. E282 shows that Kodak-selected policies
choose cap `0.0040`, which transfers to CLIC with negative mean score but weak
fixed-index selected win. CLIC-aware or all-domain fixed policies are safer.

The next implementation target is therefore not a larger all-on branch. It is a
decoder-safe selected replacement controller whose decision can be reproduced or
coarsely signaled, and whose active RVQ index cost is charged explicitly. For
EF-LIC, preserve the official representation-domain decorrelation and RVQ path,
insert HCG only as a local geometry/replacement controller after the existing
mean/scale normalization, and report fixed-index accounting from the first pilot.

## E283 Signal-Overhead Implementation Update

The selected-replacement branch now has an explicit signaling contract. A coarse
image-level mode flag is cheap enough that it does not change the GLC conclusion,
while dense tile maps consume measurable RD margin. Therefore the next GLC code
path should implement selected replacement as a coarse decoder-safe mode first:
if the replacement controller fires, send the active RVQ indices and do not also
send the replaced scalar residual bits; otherwise fall back exactly to the base
GLC scalar path.

For GLC paper-main experiments, keep two cap/controller settings: cap `0.0035`
for the balanced coarse-signaled claim, and cap `0.0030` for stricter fixed-index
or no-entropy ablations. Cap `0.0040` remains an aggressive performance-search
branch, but it must carry CLIC failure-case analysis because fixed-index selected
wins are weaker.

For EF-LIC integration, preserve the official representation-domain
decorrelation and no-entropy-coding premise. HCG-RVQ should initially be plugged
in as a fixed or coarsely signaled quantizer/replacement mode, not as a dense
entropy-like side stream. Any additional local gate map must be charged and
reported separately.

## GLC E284 Signal-Accounted Replacement Row Contract

The GLC replacement pilot now has an explicit row contract for decoder-visible selection/mode signaling. Future GLC runs that emit replacement rows should add:

--emit-replacement-rows --replacement-cap-dbpp 0.0035 --replacement-cap-dbpp-values 0.0030 0.0040 --replacement-signal-bits 1 8

Required reporting fields:

- selection_signal_bpp: image-level or coarse-mode signal cost charged to the row.
- active_scalar_bpp, active_rvq_empirical_bpp, active_rvq_fixed_bpp, and active_replacement_delta_bpp: the replacement-rate decomposition.
- selected_frac, gate_mean, active_mse_ratio, index_entropy_mean, and nonfinite counts.

Interpretation rule:

- No-signal rows are allowed as diagnostic upper bounds only.
- 1-bit or 8-bit image-signal rows are the first paper-facing coarse-controller candidates.
- Cap 0.0035 is the balanced GLC controller; cap 0.0030 is the safer fixed-index/no-entropy controller; cap 0.0040 is aggressive and must include failure analysis.

This contract should be reused when porting HCG-RVQ into EF-LIC. EF-LICs Representation-domain Decorrelation and no-entropy-coding structure should stay as the baseline backbone claim; HCG-RVQ should enter as a smarter, explicitly accounted VQ/RVQ quantizer or replacement controller, not as an uncharged hidden side channel.

## GLC/EF-LIC Controller Contract After E287

The current HCG-RVQ replacement branch should be promoted as a selected replacement mode, not as dense all-on quantization. The controller family to carry forward is:

- cap 0.0030: strict no-entropy/fixed-index candidate. It sacrifices selection rate but keeps selected empirical and fixed-index wins at 1.000000 on the E285/E286 current subset. Use this as the safest EF-LIC starting point.
- cap 0.0035: balanced paper-facing GLC controller. It keeps selected empirical win 1.000000 and selected fixed-index win 0.950000 on the E285/E286 current subset while preserving more mean gain than cap 0.0030.
- cap 0.0040: aggressive performance branch. It gives the best mean score on the current subset but admits hard CLIC-tail failures, so it must be reported with failure analysis rather than used as the only main claim.

Any decoder-visible mode/selection signal must be charged explicitly. Image-level 1-bit/8-bit signaling is currently negligible in the GLC subset, but dense spatial gates remain diagnostic unless they are compressed or predicted from decoder-available state. EF-LIC should preserve its representation-domain decorrelation/no-entropy identity and add HCG-RVQ only as a measured quantizer/replacement controller with fixed-index or explicitly signaled accounting.

## GLC Full-Training Gate After E377-E378

The first GLC branch that should move beyond short-cycle diagnostics is the q-aware `index_entropy_mean` reliability controller with a 0.02 safety margin. It is preferred over dense/all-on replacement because E377/E378 show that the useful region is a decoder-reproducible high-entropy active subset, while low-reliability states need exact scalar fallback.

Implementation requirements for the promoted branch:

1. Use q-aware high-entropy activation with the calibrated safety margin.
2. Preserve the original scalar/GLC path outside the active subset.
3. Keep the original GLC perceptual codec objective dominant; add only minimal VQ/index-rate terms needed for measurable bit accounting.
4. Evaluate checkpoints by LPIPS, DISTS, MS-SSIM, bpp, selected fraction, fixed-index tail score, codebook usage, residual-stage utilization, and nonfinite/decode consistency.
5. Treat the global high-entropy margin policy as the simple ablation and no-margin q-aware policy as a high-gain diagnostic, not the main paper branch.

This is a promotion to matched fine-tuning/full-training for GLC. EF-LIC should remain in controller-search mode until a perceptual slice/block reliability policy passes a comparable held-out gate.

