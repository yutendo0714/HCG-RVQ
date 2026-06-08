# HCG-RVQ Research Reading Notes

Date: 2026-05-29

This note records the pre-implementation reading pass for HCG-RVQ. It is meant to preserve the project assumptions before coding starts.

## 2026-06-02 Literature And Code Refresh

The project goal is unchanged after rereading `docs/prompt.txt`: HCG-RVQ should show that a hyperprior can generate local vector-quantizer geometry, not only entropy parameters. The safest paper claim is therefore:

> Hyperprior-generated local shift, scale, and orthogonal geometry can improve RVQ-based learned image compression when evaluated with matched checkpoints, feature diagnostics, and reliability control.

Current experiment context:

- The true holdout4096 audits support a prototype geometry signal: fixed HCG-H gate variants improve the 3-seed mean over HCS-RVQ, but old gate0.25 and min090 have complementary seed/image behavior.
- The main next method direction remains constrained reliability/usage control, not a free RD-trained reliability head.
- SOTA claims are not ready yet. The current evidence is a MeanScaleHyperprior/RVQ prototype claim; SOTA work should be framed as future plug-in or one carefully controlled strong-backbone experiment.

Refreshed external-source conclusions:

- [RDVQ](https://arxiv.org/abs/2604.10546) is the most important VQ-RD competitor. It is accepted as CVPR 2026 oral and uses a differentiable relaxation of the codebook distribution so entropy loss can shape the latent prior. Its [GitHub repository](https://github.com/CVL-UESTC/RDVQ) currently says code will come soon, so it should be treated as a paper/theory competitor for now, not an implementation base.
- [HVQ-CGIC](https://arxiv.org/abs/2512.07192) is the closest hyperprior-VQ competitor. It introduces hyperprior entropy modeling for VQ indices in controllable generative compression. HCG-RVQ must distinguish itself by conditioning the quantizer geometry before index selection. A paper table should explicitly separate entropy-only, HCS shift/scale, HCG-H geometry, and reliability control.
- [Adaptive LVQ](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Multirate_Neural_Image_Compression_with_Adaptive_Lattice_Vector_Quantization_CVPR_2025_paper.html) supports the broad claim that quantizer design can drive rate/domain adaptation. Its adaptation is lattice/rate/domain level; HCG-RVQ should emphasize spatial and channel-group local geometry over a shared RVQ codebook.
- [NUCQ](https://rc.signalprocessingsociety.org/conferences/icip-2023/spsicip23vid0176) is a close conditional-quantization predecessor because it learns content-adaptive non-uniform scalar quantization from hyperprior/textural information. HCG-RVQ should cite it and avoid claiming that hyperprior-conditioned quantization in general is new.
- [ProGIC](https://arxiv.org/abs/2603.02897) confirms that RVQ stages are a natural progressive compression structure. For HCG-RVQ, soft stage gates are not enough for a bitrate claim; any stage-gate claim needs deterministic decoder-known stage skipping or a bitstream-consistent entropy rule.
- [CompressAI](https://github.com/InterDigitalInc/CompressAI) remains the right infrastructure reference because it provides hyperprior models, entropy bottlenecks, pretrained/evaluation conventions, and arithmetic-coding infrastructure. Standard mean-scale hyperprior already uses conditional scalar means, so HCG-RVQ should not overclaim shift conditioning as the novelty.
- [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) remains a useful implementation reference, especially for `ResidualVQ`, grouped RVQ, k-means initialization, quantize dropout, EMA/stale-code handling, FSQ/RFSQ, SimVQ, and rotation-trick options. These are stabilization ablations, not a drop-in replacement for the paper-main HCG quantizer.
- [SimVQ](https://arxiv.org/abs/2411.02038) and its [code](https://github.com/youngsheen/SimVQ) are relevant if codebook usage or dead-code behavior limits HCG-H. Its key implementation idea is code-vector reparameterization through a learnable linear transform over a latent basis so optimization touches the codebook space more globally.
- [Rotation Trick](https://arxiv.org/abs/2410.06424) and its [code](https://github.com/cfifty/rotation_trick) are relevant if STE instability or quantization-error gradients become the bottleneck. It should be tested after the reliability/selector path is stable, because changing the gradient estimator can confound the current geometry attribution.
- Strong LIC backbones remain comparison/plug-in targets, not immediate replacements: [HPCM](https://github.com/lyq133/LIC-HPCM) for hierarchical/progressive context, [MLIC/MLIC++](https://github.com/JiangWeibeta/MLIC) for multi-reference entropy modeling, [TCM](https://github.com/jmliu206/LIC_TCM) for Transformer-CNN mixed transforms, [MambaIC](https://github.com/AuroraZengfh/MambaIC) for SSM context/backbone design, and [DCAE](https://github.com/CVL-UESTC/DCAE) for dictionary-based entropy modeling.

Updated implementation policy:

1. Keep the core HCG quantizer self-implemented. The novelty claim depends on isolating hyperprior-conditioned quantizer geometry from borrowed external recipes.
2. Use official repositories for baselines, protocol checks, and later plug-in experiments. Do not copy a full external VQ/GIC pipeline into the main method unless the ablation explicitly records the confound.
3. Promote only bitstream-consistent mechanisms into paper claims. Index entropy and deterministic stage skipping need to match the decoder-visible information.
4. Make the next paper-facing ablation table explicit: fixed/global RVQ, entropy-only/HVQ-like index prior, HCS shift/scale, no-transform control, HCG-H geometry, and constrained reliability/usage selector.
5. Use stabilization tricks in this order if needed: better checkpoint protocol and teacher split first, then SimVQ-style codebook reparameterization or rotation-trick gradient as separate ablations, then FSQ/RFSQ fallback only if RVQ remains unstable.
6. Keep CUDA experiments on `CUDA_VISIBLE_DEVICES=0` / `cuda:0`; device 1 should not be used for new runs.

## Project Position

The strongest HCG-RVQ claim should not be "the hyperprior adapts quantization" in a broad sense. That framing is vulnerable because scalar mean-conditioned quantization, non-uniform conditional quantization, spatial companding, and adaptive lattice quantization already exist.

The stronger claim is:

> The hyperprior instantiates local vector quantizer geometry: a shared RVQ codebook is transformed by content-conditioned shift, scale, and orthogonal reflections, with entropy-aware stage allocation.

In other words, the core novelty is local geometry of a shared RVQ codebook, not merely a better VQ index entropy model.

## Base Implementation: CompressAI

Primary base:

- https://github.com/InterDigitalInc/CompressAI
- https://interdigitalinc.github.io/CompressAI/models.html
- https://interdigitalinc.github.io/CompressAI/entropy_models.html
- https://interdigitalinc.github.io/CompressAI/_modules/compressai/models/google.html

Important implementation facts:

- `ScaleHyperprior` uses `g_a -> y`, `h_a(abs(y)) -> z`, `EntropyBottleneck(z) -> z_hat`, `h_s(z_hat) -> scales_hat`, then `GaussianConditional(y, scales_hat)`.
- `MeanScaleHyperprior` changes this to `h_a(y)` and `h_s(z_hat) -> gaussian_params`, split into `scales_hat, means_hat`.
- `GaussianConditional.quantize(inputs, mode, means)` subtracts `means` before rounding and adds them back for dequantization. This means standard mean-scale hyperprior already conditions scalar quantization centers.
- CompressAI training uses primary RD loss plus `aux_loss()` for entropy bottleneck quantiles. Real entropy coding requires `update()` or `compressai.utils.update_model`.
- For HCG-RVQ models, do not double-count `GaussianConditional` y likelihood and VQ index likelihood. In RVQ variants, rate should be:
  - `R_z`: hyper latent likelihood from `EntropyBottleneck`
  - `R_y`: categorical VQ/RVQ index entropy proxy or actual coded bits

Recommended project approach:

- Keep this repository independent.
- Use CompressAI as a dependency and reference implementation.
- Copy/subclass the MeanScaleHyperprior structure only where needed to expose `hyper_features`, `mu_q`, `log_s_q`, `v`, `gate`, and index logits.

## Rate Budget Reality Check

For a latent stride of 16, input pixel count is 256 times the y-grid count. If `M=320`, `group_size=16`, and `num_stages=4`, then:

```text
symbols_per_pixel = (M / group_size) * num_stages / 16^2
                  = (320 / 16) * 4 / 256
                  = 0.3125 symbols / pixel
```

At 0.2 bpp, the index model must average only:

```text
0.2 / 0.3125 = 0.64 bits / symbol
```

That is very strict. Initial pilots should be smaller:

- `group_size=32`
- `num_stages=1 or 2`
- `codebook_size=128 or 256`

The originally suggested `group_size=16, L=4, K=256` can be tried later, but it may be too expensive unless the entropy model is extremely sharp or stage skipping truly avoids coding skipped indices.

## Stage Gate Caveat

A soft stage gate reduces reconstruction contribution, but it does not reduce bitrate if all stage indices are still transmitted.

For stage gate to be a compression contribution, one of the following must be true:

- hard skip mask is generated from `z_hat`, so the decoder knows which stages are absent and no indices are coded for skipped stages;
- a differentiable/annealed proxy trains gate usage, then evaluation thresholds to a deterministic skip mask;
- the `R_y` loss is explicitly weighted by gate usage and the final bitstream follows the same skip rule.

Otherwise, "average stage usage" is only a reconstruction statistic, not a bitrate statistic.

## Related Work: Directly Relevant

### Ballé 2018 Scale Hyperprior

Source:

- https://arxiv.org/abs/1802.01436
- CompressAI implementation: `ScaleHyperprior`

Role:

- Foundation of side information in LIC.
- HCG-RVQ reuses the hyperprior path, but changes what `h_s(z_hat)` controls.

HCG-RVQ distinction:

- Ballé predicts entropy scales for scalar latents.
- HCG-RVQ predicts vector quantizer parameters and index priors.

### Minnen 2018 Mean-Scale / Joint Autoregressive Hierarchical Priors

Source:

- https://arxiv.org/abs/1809.02736
- CompressAI implementation: `MeanScaleHyperprior`, `JointAutoregressiveHierarchicalPriors`

Role:

- Important nuance: mean-conditioned scalar quantization already uses hyper/context information as a dequantization offset.

HCG-RVQ distinction:

- The novelty should be described as local vector quantizer geometry, not simply "hyperprior shifts quantization."

### HVQ-CGIC

Source:

- https://arxiv.org/abs/2512.07192

Key point from abstract:

- Static/global VQ index distributions are non-adaptive.
- HVQ-CGIC introduces hyperprior entropy modeling for VQ indices in controllable generative image compression.

HCG-RVQ distinction:

- HVQ-CGIC conditions the entropy model over VQ indices.
- HCG-RVQ conditions the quantizer geometry before index selection.

Required ablation:

- hyperprior-conditioned entropy only
- entropy + shift/scale RVQ
- entropy + shift/scale + Householder geometry

### RDVQ

Source:

- https://arxiv.org/abs/2604.10546
- https://github.com/CVL-UESTC/RDVQ

Key point:

- RDVQ enables end-to-end RD optimization for VQ via differentiable relaxation of the codebook distribution.
- It also develops autoregressive entropy modeling and test-time rate control.
- As of this reading pass, arXiv says the code will be available at the GitHub URL; it is not a reliable implementation base yet.

HCG-RVQ distinction:

- RDVQ addresses differentiable VQ RD optimization.
- HCG-RVQ addresses local content-adaptive quantizer geometry generated by the hyperprior.

Implementation implication:

- A plain hard argmin CE index loss trains the index prior, but it does not fully make index selection differentiably rate-aware. RDVQ-style relaxation may be needed in a later phase if `R_y` does not shape quantizer behavior enough.

### ProGIC

Source:

- https://arxiv.org/abs/2603.02897

Key point:

- Compact progressive generative image compression with RVQ.
- RVQ stages yield coarse-to-fine reconstruction and progressive bitstream.

HCG-RVQ distinction:

- ProGIC is progressive RVQ with a lightweight generative codec.
- HCG-RVQ adds hyperprior-conditioned local geometry and deterministic/content-adaptive stage allocation.

Required comparison angle:

- If stage gating is a claim, HCG-RVQ must show actual coded stage reduction, not just soft weights.

### LVQAC

Source:

- https://arxiv.org/abs/2304.12319

Key point:

- Replaces scalar quantization with lattice vector quantization.
- Adds spatially adaptive companding to adapt to source statistics.
- Designed to embed into existing end-to-end image compression systems.

HCG-RVQ distinction:

- LVQAC uses lattice VQ plus spatial companding.
- HCG-RVQ uses free/shared RVQ codebooks and local affine/orthogonal transformation predicted from hyper latents.

### Adaptive LVQ

Source:

- https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Multirate_Neural_Image_Compression_with_Adaptive_Lattice_Vector_Quantization_CVPR_2025_paper.pdf

Key point:

- Adaptive LVQ solves rate adaptation by scaling lattice basis vectors and domain adaptation via learned invertible linear transformations between domains.
- It argues quantizer design can be a direct adaptation mechanism.

HCG-RVQ distinction:

- Adaptive LVQ adapts rate/domain at codec or domain level.
- HCG-RVQ adapts spatially and channel-wise per image location through hyperprior-generated local transforms.

### NUCQ

Source:

- https://rc.signalprocessingsociety.org/conferences/icip-2023/spsicip23vid0176

Key point:

- Non-uniform conditional quantization conditioned on image textural/hyperprior information.
- Learns a parameterized quantile model from hyperprior information.
- Reports BD-rate gains over LIC baselines.

HCG-RVQ distinction:

- NUCQ is conditional non-uniform scalar quantization / quantile modeling.
- HCG-RVQ is vector codebook geometry: grouped RVQ codewords are locally shifted, scaled, and orthogonally transformed.

This paper must be included in related work because it is one of the closest "hyperprior-conditioned quantizer" predecessors.

## VQ/RVQ Stabilization and Fallbacks

### vector-quantize-pytorch

Source:

- https://github.com/lucidrains/vector-quantize-pytorch

Relevant implementation ideas:

- `ResidualVQ`
- `GroupedResidualVQ`
- k-means initialization
- shared codebooks
- stochastic code sampling
- codebook dimension projection
- cosine similarity
- stale code expiration
- orthogonal regularization
- SimVQ
- FSQ / ResidualFSQ
- LFQ

Project decision:

- Do not directly depend on it for the main HCG-RVQ module because compression-specific rate accounting, index entropy modeling, and stage gating need tight control.
- Use it as a reference for stabilization tricks if dead codes become severe.

### SimVQ

Sources:

- https://arxiv.org/abs/2411.02038
- https://github.com/youngsheen/SimVQ

Key point:

- Freezes or reparameterizes the codebook through a learned linear projection over latent basis vectors.
- Improves codebook utilization / reduces collapse in tokenizers.

Implementation implication:

- Not Phase 1.
- Add as a fallback if Global RVQ or HCS-RVQ has high dead-code ratio.

### FSQ

Source:

- https://arxiv.org/abs/2309.15505

Key point:

- Removes explicit codebook, commitment loss, EMA, and collapse issues by finite scalar levels in low dimensions.

Implementation implication:

- Not the main path, because the central claim is vector quantizer geometry.
- Useful fallback if RVQ stability dominates the project.

### Rotation Trick / DiVeQ

Sources:

- https://arxiv.org/abs/2410.06424
- DiVeQ referenced in vector-quantize-pytorch documentation

Key point:

- Improve VQ gradient estimation beyond simple STE.

Implementation implication:

- Keep simple STE first.
- Consider rotation trick or RDVQ-style relaxation only if optimization stalls.

## Strong Backbones and Later Integration

### ELIC

Source:

- https://arxiv.org/abs/2203.10886

Role:

- Uneven channel grouping and space-channel context are highly relevant to HCG-RVQ grouping.
- Not the first implementation base because it makes isolating quantizer effects harder.

### MLIC / MLIC++

Sources:

- https://arxiv.org/abs/2211.07273
- https://github.com/JiangWeibeta/MLIC

Role:

- Strong multi-reference entropy modeling.
- Later comparison/backbone candidate.

### HPCM

Sources:

- https://arxiv.org/abs/2507.19125
- https://github.com/lyq133/LIC-HPCM

Role:

- Hierarchical progressive coding schedule and progressive context fusion.
- Highly relevant to future stage-context entropy model:

```text
p(k_l | z_hat, k_0, ..., k_{l-1})
```

Implementation note:

- Its GitHub notes include a custom arithmetic coder path. Useful later for real bitstream work.

### MambaIC

Sources:

- https://arxiv.org/abs/2503.12461
- https://github.com/AuroraZengfh/MambaIC

Role:

- Strong high-performance SSM-based LIC model.
- Later plug-in candidate only after HCG-RVQ is validated on MeanScaleHyperprior.

### DCAE

Sources:

- https://arxiv.org/abs/2504.00496
- paper-related repo appears to be https://github.com/LabShuHangGU/DCAE rather than `CVL-UESTC/DCAE`

Role:

- Dictionary-based cross-attention entropy model.
- Important comparison because it also uses dictionary/codebook-like priors, but on the entropy modeling side.

HCG-RVQ distinction:

- DCAE uses a dictionary to predict latent entropy parameters.
- HCG-RVQ uses a shared RVQ codebook as the quantizer and changes its local geometry.

## Implementation Decisions After Reading

Initial model family:

1. `MeanScaleHyperpriorScalar`
2. `MeanScaleHyperpriorGlobalRVQ`
3. `MeanScaleHyperpriorHCSRVQ`
4. `MeanScaleHyperpriorHCSRVQIndexPrior`
5. `MeanScaleHyperpriorHCGRVQH`
6. `MeanScaleHyperpriorHCGRVQHGate`

Initial defaults should be conservative:

```yaml
N: 192
M: 320
group_size: 32
num_stages: 2
codebook_size: 256
beta_commit: 0.25
scale_min: 0.05
scale_max: 10.0
index_prior: hyper_only
stage_context: false
stage_gate: false
```

Metrics to log from the first RVQ run:

- `bpp_total`
- `bpp_y_index`
- `bpp_z`
- `mse`
- `psnr`
- `ms_ssim`
- `commit_loss`
- `latent_quant_mse`
- `codebook_perplexity`
- `dead_code_ratio`
- `stage_entropy`
- `stage_usage`
- `s_q_mean/min/max` for HCS/HCG
- `householder_v_norm_mean` for HCG-H

Important training warning:

- `R_y = CE(index_logits, hard_indices)` is a valid entropy-model training proxy, but hard argmin indices do not provide a direct differentiable rate gradient through assignment. If the encoder/quantizer fails to become entropy-friendly, add soft assignment, Gumbel/temperature relaxation, RDVQ-style distribution relaxation, or stage-wise entropy regularization.

## Current Go/Pivot Criteria

Go signs:

- HCS-RVQ improves latent quantization MSE over Global RVQ.
- HCS-RVQ improves total RD after including `R_z + R_y`.
- HCG-RVQ-H improves over HCS-RVQ under the same index prior.
- Index prior lowers estimated `R_y` without increasing `R_z` too much.
- Hard/annealed gate reduces actual transmitted indices.

Pivot signs:

- HCS improves distortion but total bpp worsens badly.
- Householder adds no measurable benefit over shift/scale.
- Most codebook entries die even with conservative `K/L/G`.
- Gate becomes always-on or saves no coded rate.

If pivot is needed, the backup story is:

> Hyperprior-Conditioned Adaptive Residual Vector Quantization with entropy-aware stage allocation.

This weaker story can still be useful if shift/scale and gating work but Householder geometry does not.

## 2026-05-30 Re-reading: Paper and Official-Code Implications

This pass re-checked the closest papers and public repositories from the viewpoint of the current evidence: HCS/index-prior is solid, HCG-H geometry is promising but seed/checkpoint-sensitive, and the old/min090 selector gives headroom that still has to become a single-checkpoint controller.

### RDVQ

Sources:

- Paper: https://arxiv.org/abs/2604.10546
- Official repository: https://github.com/CVL-UESTC/RDVQ

Reading outcome:

- RDVQ is the most relevant VQ-RD optimization competitor because it connects VQ code selection to rate-distortion training through a differentiable distribution relaxation.
- The repository currently appears to be a placeholder for the CVPR 2026 work rather than an implementation-ready codebase. Treat it as a required paper comparison and future implementation reference, not as a dependency for the next HCG experiments.

HCG implication:

- RDVQ strengthens the need to distinguish entropy-aware VQ training from the HCG claim.
- If the current hard-index prior cannot make code selection rate-friendly enough, RDVQ-style relaxed index distributions are the first fallback to study.
- The immediate main line should not pivot to RDVQ. It should keep the HCG novelty: hyperprior-generated local quantizer geometry.

### HVQ-CGIC

Sources:

- Paper: https://arxiv.org/abs/2512.07192

Reading outcome:

- HVQ-CGIC is the closest conceptual neighbor for hyperprior-conditioned VQ index entropy modeling.
- Its paper framing reinforces the same boundary already written in `prompt.txt`: entropy modeling of VQ indices is not enough to claim HCG novelty.

HCG implication:

- The required ablation set remains:
  - entropy-only / index-prior RVQ,
  - HCS shift-scale RVQ,
  - HCG-H geometry RVQ,
  - reliability-controlled geometry.
- If no official repository is available, implement the entropy-only comparison locally using the existing HCS/index-prior infrastructure.

### Adaptive LVQ / MLVIC

Sources:

- Paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Multirate_Neural_Image_Compression_with_Adaptive_Lattice_Vector_Quantization_CVPR_2025_paper.pdf
- Repository: https://github.com/marmotlab/MLVIC

Reading outcome:

- Adaptive LVQ is important because it shows that vector quantizer geometry/adaptation is a recognized direction in LIC, not an artificial problem invented by this project.
- The repository is useful for multirate LVQ framing and evaluation conventions.

HCG implication:

- The distinction should be stated sharply:
  - Adaptive LVQ adapts lattice vector quantization for rate/domain behavior.
  - HCG-RVQ uses the hyperprior to generate spatial/channel-wise transformations of a shared residual VQ codebook.
- This is a strong related-work anchor, but it does not replace the need for HCG-specific geometry ablations.

### DCAE

Sources:

- Repository: https://github.com/CVL-UESTC/DCAE
- Main model file inspected: https://raw.githubusercontent.com/CVL-UESTC/DCAE/main/models/dcae.py

Reading outcome:

- DCAE is a strong modern LIC baseline around dictionary-based entropy modeling and CompressAI-compatible `compress` / `decompress` paths.
- Its code reinforces a practical lesson: paper-facing methods should have a single codec path with explicit entropy coding behavior, not only a retrospective evaluation selector.

HCG implication:

- DCAE should be treated as a future strong-backbone or strong-entropy comparison.
- The current HCG-RVQ work should first prove the quantizer-geometry mechanism on the simpler MeanScaleHyperprior family, then later test whether the same geometry controller improves a DCAE/HPCM/MLIC-like backbone.
- Do not move to DCAE before the single-checkpoint reliability controller is understood; that would mix backbone gains with quantizer-geometry gains.

### VQ Stability Tooling: SimVQ, FSQ, vector-quantize-pytorch, Rotation Trick

Sources:

- SimVQ: https://github.com/youngsheen/SimVQ
- vector-quantize-pytorch: https://github.com/lucidrains/vector-quantize-pytorch
- Rotation Trick: https://github.com/cfifty/rotation_trick

Reading outcome:

- SimVQ targets representation collapse by reparameterizing the codebook with a linear layer.
- FSQ-style and lookup-free variants are useful fallback quantizers when codebook learning itself is the bottleneck.
- The Rotation Trick is a gradient-estimator improvement for VQ, and is relevant if STE limits geometry learning.
- `vector-quantize-pytorch` is a broad implementation reference for residual VQ, FSQ, LFQ, codebook usage tricks, and rotation-trick-style options.

HCG implication:

- The current failure mode is not primarily dead-code collapse; the stronger bottleneck is reliability/checkpoint sensitivity of the geometry gate.
- Therefore these tools are not the next main action.
- Keep them as second-line ablations if stage expansion, stronger codebooks, or unfrozen training reintroduce collapse or unstable VQ gradients.

### Strong LIC Backbones: HPCM, MLIC/ELIC, MambaIC

Sources:

- HPCM: https://github.com/lyq133/LIC-HPCM
- MLIC / MLIC++: https://github.com/JiangWeibeta/MLIC
- MambaIC: https://github.com/AuroraZengfh/MambaIC

Reading outcome:

- These repositories are important for strong entropy/context baselines and for later plug-in experiments.
- They are not the cleanest first place to prove HCG because their backbone/context gains can hide whether the quantizer geometry itself helped.

HCG implication:

- Use them in the paper plan as strong-backbone integration after the simple-family claim is stable.
- The first strong-backbone experiment should ask: does adding the HCG reliability-controlled quantizer strategy improve an already strong model over its own baseline?
- This is a safer SOTA-facing claim than claiming the current MeanScaleHyperprior prototype beats every modern LIC system outright.

## Updated Research Decision

The re-reading supports the current pivot rather than overturning it.

1. Keep the main novelty as hyperprior-conditioned quantizer geometry.
2. Keep HCS/index-prior as the stable mechanistic baseline.
3. Treat old gate0.25 as the safest current single-checkpoint HCG-H result.
4. Treat min090 and old/min090 selection as evidence that reliability control has real headroom, not as the final method row.
5. Make the next implementation target a unified single-checkpoint reliability controller.
6. After that, compare against entropy-only HVQ-style, RDVQ-style differentiable VQ, Adaptive-LVQ-style quantizer adaptation, and strong LIC backbones.

Progress assessment:

- The project is aligned with the `prompt.txt` goal and the novelty still survives the related-work/code check.
- It is not yet a finished international-conference claim because the strongest selector evidence is still multi-checkpoint.
- The direction is promising because the current analyses have already separated several confounders: stale artifacts, CUDA/device provenance, checkpoint choice, validation/reporting split leakage, entropy-only vs geometry effects, and image-level reliability.

## 2026-06-03 SOTA/Backbone Plug-in Policy After E151

The E151 direct evaluation changes the plug-in priority. It confirms that the strongest current improvement is not raw fixed HCG and not continuous geometry shrinkage; it is a state-preserving HCS/HCG branch controlled by a geometry reliability statistic.

Practical implication:

- Controlled-evidence lane: keep proving the quantizer-geometry claim on the current simple backbone, because it isolates the novelty and gives clean ablations against Global RVQ, HCS, entropy-only/index-prior controls, fixed HCG, and branch-controlled HCG.
- Method-strengthening lane: implement the branch/fallback as a real codec mechanism, first in the current codebase. The branch can be signaled with a tiny side bit or made decoder-reproducible if the reliability proxy becomes strong enough.
- SOTA/backbone lane: start with scouting and adapter planning, but do not spend the largest GPU budget on a raw HCG transplant. The first meaningful plug-in experiment should be "does the state-preserving HCG branch improve a strong backbone over its own baseline?" rather than "does the prototype beat every SOTA model?"

This answers the scale concern. Small smoke tests and ablations are not enough by themselves, because large backbones can expose new interactions. But starting with a large SOTA clone before the branch mechanism is fixed would make failures ambiguous. The right two-track plan is to keep the controlled evidence moving while preparing a scoped plug-in experiment whose mechanism is already supported by E151.

## 2026-06-03 Official Repo Clone Pass After E152

I cloned the official or paper-associated repositories needed for the first SOTA/backbone scout:

- DCAE: `third_party/DCAE`
- MambaIC: `third_party/MambaIC`
- LIC-HPCM: `third_party/LIC-HPCM`
- RDVQ: `third_party/RDVQ`

The implementation reading supports the same two-track policy. DCAE exposes a clean `g_a -> h_a/h_s -> sliced y_hat -> g_s` path and has train/eval/compress scripts plus pretrained links, so it is the best first forward-only plug-in smoke. MambaIC has a similar slice assembly path, but setup is more fragile because of VMamba/selective-scan dependencies. HPCM is very relevant for future RVQ stage-context entropy modeling, but its actual codec path is deeply progressive, so it should be used first as design guidance. RDVQ is currently a paper/claim comparison rather than an executable dependency because the official README says code will come soon.

Research implication:

- Do not wait until the end to touch SOTA/backbones.
- Do not port raw fixed HCG first.
- Port the E151/E152 state-preserving branch interface first, then ask whether it improves DCAE/MambaIC/HPCM over their own baselines.
- Keep RDVQ/HVQ-CGIC/Adaptive-LVQ as paper-facing comparison axes: entropy-aware VQ training, hyperprior-conditioned index entropy, and adaptive quantizer geometry respectively.



## 2026-06-03 DCAE Plug-in Smoke Reading Update

The DCAE implementation is now confirmed as the best first official-backbone plug-in target. Its forward path exposes a clean quantizer boundary: analysis transform `g_a`, hyper analysis `h_a`, hyper-derived scales/means via `h_z_s1` and `h_z_s2`, per-slice `y_hat` assembly, and synthesis transform `g_s`. E153 uses exactly that boundary by feeding `[latent_scales, latent_means]` into the local HCG adapter and decoding the adapter-produced `y_hat` through DCAE `g_s`.

The smoke result is intentionally modest but useful. It confirms compatibility and finite tensors, not quality. The next DCAE-facing experiments should therefore be ordered as follows:

1. Reproduce a DCAE baseline with official pretrained weights or a controlled local training recipe.
2. Insert the HCS/HCG state-preserving branch at the `y_hat -> g_s` boundary, not a raw continuous Householder shrinker.
3. Compare the modified backbone against the same DCAE backbone over itself, with bpp/PSNR/MS-SSIM/RD, per-image tails, checkpoint choice, and feature-distribution diagnostics.

This is the right response to the large-scale concern. Yes, a method that looks good in short ablations can fail in a large SOTA backbone. E153 starts checking that risk early, while still keeping the controlled-evidence lane clean enough for an international-conference claim.


## 2026-06-03 E154 SOTA/Backbone Reproduction Audit Update

The latest repo audit changes the SOTA/backbone plan from a vague future direction into an executable sequence. DCAE remains first because the official README clone target and the existing local clone are commit-identical, and because its forward/eval path exposes a clean latent boundary for HCG branch insertion. HPCM and MambaIC remain highly relevant, but should follow after DCAE reproduction is stable. RDVQ remains a must-cite VQ competitor, but its local README still marks code as forthcoming.

Recommended experiment order:

1. DCAE official baseline reproduction with one low-rate MSE checkpoint.
2. DCAE same-backbone HCS/HCG state-preserving branch comparison.
3. Full per-image diagnostics: RD tails, wins, checkpoint choice, feature distributions, codebook usage, side-bit accounting, and nonfinite rows.
4. HPCM and MambaIC comparison or plug-in after DCAE establishes the boundary and analysis protocol.
5. RDVQ paper comparison now, executable comparison when code becomes available.

This supports the two-lane strategy. The controlled-evidence lane makes the quantizer-geometry claim clean. The SOTA lane checks whether that claim survives in a stronger backbone, but it should do so as a same-backbone self-improvement claim first.


## 2026-06-03 E155 DCAE Pretrained Baseline Update

The DCAE lane is now beyond static reading. The official low-rate MSE checkpoint has been downloaded and evaluated on Kodak with GPU0 only. This makes DCAE the active first SOTA/backbone target rather than just a planned target.

Important interpretation:

1. E155 is a same-backbone baseline anchor, not a claim that HCG-RVQ beats DCAE.
2. Future plug-in runs must reuse the same checkpoint, paths, metric code, and per-image diagnostics.
3. The paper-facing SOTA claim should initially be self-improvement: DCAE plus HCS/HCG branch improves DCAE itself at matched rate conditions.
4. HPCM/MambaIC remain later comparison or plug-in targets after DCAE establishes the protocol.

This is the practical compromise between small controlled ablations and large bold experiments: start bold enough to touch official pretrained SOTA code, but keep the comparison narrow enough to be scientifically interpretable.


## 2026-06-03 VQ/GIC Plug-in Reassessment: GLC vs EF-LIC

I reread `docs/prompt.txt` and the local PDFs `docs/2512.20194v1.pdf` and `docs/2605.23323v1-2.pdf`, then inspected the official clones under `third_party/GLC` and `third_party/EF-LIC`.

The prompt's core claim is specifically VQ/RVQ quantizer geometry: the hyperprior should generate local shift, scale, and geometry for the quantizer itself. This means that a VQ/RVQ-based generative image compression method is a more natural SOTA-facing plug-in target than a strong non-VQ LIC backbone. DCAE remains useful as a strong-backbone sanity check, but it should not be the main proof of the VQ-improvement story.

### GLC

Sources:

- Paper: `docs/2512.20194v1.pdf`
- Official repo: `third_party/GLC`, `https://github.com/jzyustc/GLC`

Reading outcome:

- GLC is accepted at CVPR 2024 and the repository provides pretrained-test scripts.
- The paper compresses in the latent space of a generative VQ-VAE and reports ultra-low-bitrate perceptual results, including less than 0.04 bpp on natural images and a 45% bitrate saving over MS-ILLM at the same FID on CLIC2020.
- The implementation uses a VQGAN/VQ-VAE latent space, a categorical hyper module for `z`, and a Gaussian/prior-style transform-coding path for `y`.
- In code, `GLC_Image.test()` follows `vqgan.encoder(x) -> enc -> hyper_enc -> z_vq -> hyper_dec/y_prior -> rounded y_hat -> dec -> vqgan.generator`.
- Therefore GLC is highly relevant as a generative latent/VQ compression baseline, but its main compression-side `y` quantizer is not already an RVQ module. HCG-RVQ could be inserted, but it would be a larger method surgery: replace the `y` rounded/Gaussian prior path with HCS/HCG-RVQ, or modify the VQGAN/z categorical VQ path.

Risk:

- Good official benchmark target, but less direct as a "replace the RVQ quantizer with HCG-RVQ" experiment.
- Training appears to be multi-stage: VQ-VAE stage, latent transform stage, and code-prediction supervision stage. Reproducing full training is heavier than a small adapter smoke.

### EF-LIC

Sources:

- Paper: `docs/2605.23323v1-2.pdf`
- Official repo: `third_party/EF-LIC`, `https://github.com/SevenCTHU/EF-LIC`

Reading outcome:

- EF-LIC is marked accepted by ICML 2026 on arXiv/repo and is explicitly VQ/RVQ based.
- The paper removes entropy coding by using unconstrained VQ/RVQ indices with near-maximum-entropy usage plus representation-domain latent decorrelation.
- The evaluation protocol is very aligned with a paper-facing low-bitrate GIC story: Kodak, Tecnick, DIV2K, and CLIC2020 with LPIPS/DISTS vs BPP, plus PSNR for completeness and latency.
- The code has a clean RVQ insertion boundary. In `EF_LIC.py`, each `y_slice` is normalized by context-predicted `mean` and `scale`, then passed through `ResidualVectorQuantizeDropInfer`; the decoded vector is de-normalized and used for the next autoregressive slice.
- This is conceptually very close to HCS/HCG-RVQ: `hyper/context -> mean/scale -> RVQ -> inverse transform`, except EF-LIC does not use entropy coding and uses fixed-length packed RVQ indices.

Risk:

- The released repository is inference-only. It provides a pretrained inference checkpoint link, but no training loop.
- A paper-level claim "EF-LIC + HCG-RVQ improves EF-LIC after retraining under the same recipe" would require either official training code, author contact, or a local training reconstruction from the paper.

### Decision

For the HCG-RVQ paper story, EF-LIC is the better first VQ-SOTA target in principle because it already uses RVQ as the compression bottleneck. It lets us ask the clean question:

> Can hyperprior/context-conditioned quantizer geometry improve an already strong RVQ-based generative image codec over its own RVQ quantizer?

GLC is still valuable, but more as a generative latent/VQ benchmark and second plug-in target. It is not as clean for the immediate quantizer-replacement claim because replacing GLC's compression-side quantization would alter more of the codec.

Recommended order:

1. Keep the controlled-evidence lane active: current HCS/HCG branch evidence remains the cleanest mechanism proof.
2. Start EF-LIC reproduction immediately: download official checkpoint, run Kodak all 5 `force_ind` points on GPU0, and record BPP/PSNR/LPIPS/DISTS/latency plus nonfinite checks.
3. Build an EF-LIC forward-only HCG adapter smoke at the normalized `y_slice` RVQ boundary. First goal is finite tensors and feature diagnostics, not quality.
4. Decide the training path after the smoke:
   - If official training code appears or can be obtained, retrain EF-LIC baseline and EF-LIC+HCG under matched conditions.
   - Otherwise, implement a local minimal training reconstruction and label the result as a reconstructed-protocol plug-in, not an official reproduction.
5. Use GLC as the next benchmark after EF-LIC: reproduce pretrained image compression, then inspect whether the least-invasive HCG insertion is `z_vq`, the latent `y` quantizer, or a two-state branch around the transform-coded latent path.

Paper implication:

- The strongest eventual claim is not "HCG-RVQ beats every SOTA codec outright."
- The safer and stronger claim is "HCG-RVQ improves RVQ-based generative LIC when plugged into a strong RVQ backbone under matched conditions," with EF-LIC as the first target.
- DCAE should remain a strong-backbone side lane, not the main VQ novelty proof.

## 2026-06-03 GLC/EF-LIC Plug-in Reading Update

The paper/code reading now supports a clear split. EF-LIC is the direct RVQ plug-in target: it normalizes each y-slice by context-predicted mean/scale and applies projected-space RVQ, which matches HCG-RVQ's quantizer-geometry claim. GLC is a strong generative low-bitrate benchmark and risk hedge, but its main compression latent uses masked scalar rounding with Gaussian bit estimation; the explicit VQ is mostly in the hyper-side `z_vq`. Therefore GLC should be reproduced and tracked, but the first conference-facing HCG-RVQ replacement experiment should stay on EF-LIC.

## 2026-06-03 Pretrained vs Matched Training Policy

For EF-LIC and GLC, the research path should be: official pretrained reproduction first, insertion/bitstream smoke second, matched training or matched fine-tuning third. Pretrained-only modifications are useful diagnostics but should not be the final international-conference evidence. EF-LIC's direct RVQ bottleneck makes it ideal for HCG-RVQ replacement, while GLC needs a split plan: a lightweight `z_vq` geometry smoke and a larger `y`-path HCG-RVQ replacement that will require Stage II/III-style retraining.




## 2026-06-03: GLC Evaluation Provenance

E164 verified that the instrumented GLC path used for feature auditing exactly reproduces official `GLC_Image.test()` on Kodak24 (`q=0..3`) with zero `x_hat` and bit differences. This makes GLC suitable as a parallel main track for HCG-RVQ, provided final claims use matched retraining/fine-tuning rather than only pretrained forward modifications.


## 2026-06-03: GLC Insertion Decision

For GLC, `z_vq` is useful for a quick explicit-VQ smoke but is not the strongest HCG-RVQ novelty point. The main paper-relevant path is to replace the scalar-rounded `y` residual coding in `forward_four_part_prior()` with a decoder-context-conditioned RVQ and index prior. Final claims should use matched Stage II/III training or fine-tuning, with full Stage I scratch as optional later robustness.


## 2026-06-03: Prior HCG Evidence For EF-LIC/GLC Transfer

E166 proves that GLC main `y` residual coding can be wrapped at `forward_four_part_prior()` with exact official reproduction. E167 consolidates prior HCG experiments into the transfer rule: port the state-preserving branch/fallback mechanism, not raw always-on geometry or continuous gate shrinkage. This rule applies to both EF-LIC projected-RVQ and GLC `y`-path HCG-RVQ.


## 2026-06-03: GLC y-Path Residual Audit After E166

E168 shows that the GLC main `y` path should not be ported as a dense always-on RVQ replacement. The scalar residual stream is mostly zero after rounding, but a few early-part channel groups have very large tails. This is exactly the kind of nonstationary local distribution that HCG-RVQ is meant to address, but only if the branch is sparse or state-preserving.

Transfer rule update: for GLC, start with a part/group-aware HCS-like fallback plus HCG-RVQ active state inside the E166 `forward_four_part_prior()` wrapper. Use predicted scale, part id, group id, local residual-energy proxies, and later HCG geometry strength as reliability features. Image-level DISTS difficulty is useful for reporting tails, but it is not clean enough to be the only selector.

This keeps GLC as a main low-bitrate generative target while respecting the prior HCG lesson from E151/E167: active geometry should be deployed as a state-preserving branch, not as continuous suppression or dense always-on replacement.


## 2026-06-03: GLC Sparse Active-State Branch After E169

E169 validates a practical GLC branch scaffold for HCG-RVQ. A static decoder-known subset, active parts `[0, 1, 2]` and active groups `[1, 7, 10, 15]`, preserves official GLC exactly while covering most of the residual energy. This means GLC does not need to start with a dense replacement of all scalar-rounded `y` residuals.

Implementation rule update: the next active branch should replace only the selected subset first. The fallback remains original scalar rounding. The first active candidate can be HCS-RVQ or HCG-RVQ with local shift/scale and a small codebook; later variants can add Householder geometry and index-prior bit accounting. This gives a controlled low-bitrate generative benchmark that is still faithful to the HCG-RVQ thesis.

## 2026-06-03: GLC Tail Quantizer Decision After E170/E171

E170/E171 refine the GLC active-state design. A shared active codebook fails badly, while part/group-local codebooks reduce residual MSE on the sparse heavy-tail subset. This is a direct empirical match to the HCG-RVQ claim that quantizer geometry should be local and hyperprior/context conditioned rather than global.

The rate side is just as important. Part/group K=8 gives residual-MSE headroom across q0-q3, but its index rate is higher than scalar active bpp. K=2 RVQ stages help at q0/q1 but do not handle q3 tails. Therefore, the GLC branch should combine local geometry with an index prior and rate-aware codebook/stage selection. The final GLC paper row must come from matched training or fine-tuning, not from Kodak leave-one-image-out diagnostic codebooks.

## 2026-06-04: GLC Integrated Branch Lesson After E173/E174

E173/E174 show why GLC must be treated as a generative/perceptual compression target, not just a latent residual-MSE target. K=8 part/group VQ on the sparse active subset improves PSNR and active residual MSE, proving that the selected subset matters. But DISTS worsens almost everywhere and bpp rises, so a fixed diagnostic VQ is not the final method.

The transfer rule is now sharper: for GLC, HCG-RVQ needs a trainable active state with decoder/perceptual-aware loss, index prior, and reliability fallback. K=4 is a useful negative ablation showing that low rate without enough active capacity damages q2/q3. This supports the broader paper claim that the contribution is not replacing scalar rounding with any VQ; it is hyperprior/context-conditioned local quantizer geometry with controlled deployment.

## 2026-06-04: GLC Branch After Decoder-Aware Tail VQ Diagnostic

E175/E176 add an implementation lesson to the GLC reading: because GLC is a generative/perceptual codec, the active HCG-RVQ branch cannot be selected by latent residual MSE alone. A tiny DISTS-aware trainable-codebook diagnostic substantially reduces the DISTS damage from fixed VQ, while sometimes worsening active residual MSE. This matches the paper-level hypothesis that hyperprior/context should generate a decoder-aware local quantizer, not merely a better Euclidean residual quantizer.

For GLC, the next implementation should move from Kodak-overfit diagnostics to a Stage-II-style training split with scalar fallback, part/group active branch, q-dependent stages, index-prior accounting, and a reliability selector. The E173 fixed VQ result and E175 decoder-aware result should be used as motivation/ablation evidence, not as final rows.

## 2026-06-04: GLC Split-Train Active Branch Evidence

E177/E178 show that the GLC active branch can be trained on OpenImages crops and still transfer some gains to Kodak. This strengthens the decision to keep GLC as a main track, but it also confirms that a final claim needs reliability fallback and rate modeling. The strongest paper framing is now: HCG-RVQ learns when and how to locally replace or augment scalar residual coding under decoder-safe context, rather than globally replacing GLC quantization with a fixed residual VQ.

## 2026-06-04: GLC Selector/Loss Update After E179-E183

The GLC route remains useful, but its active branch should now be treated as a metric-sensitive reliability-control problem. OpenImages-trained sparse active codebooks improve LPIPS/PSNR/MS-SSIM broadly, but DISTS improves only on a small subset. This reinforces the HCG-RVQ framing: the contribution should be local context-conditioned quantizer geometry plus controlled deployment, not a global VQ replacement.

For the next implementation, prioritize deployable selector features available at the GLC y-prior boundary and an index-prior/bit-aware loss. Avoid paper claims based on selectors that require image-level true metrics like `base_dists`.

## E184 EF-LIC Selector Lesson

E184 connects the EF-LIC implementation track back to the HCG-RVQ thesis. The strongest current EF-LIC result is not the raw active branch, but selective use of a weak geometry mode. DISTS oracle headroom exists for every force, yet decoder-safe LOOCV only remains useful for force0. This is consistent with the broader HCG-RVQ evidence: geometry changes should be state-preserving and reliability-controlled; stronger/unconditional geometry is fragile.

For the next EF-LIC migration, prioritize:

- force0-style weak projected geometry first;
- decoder-safe reliability features from hyperprior/context statistics;
- exact original fallback;
- explicit side-bit accounting if any encoder-only selector is used;
- full-training readiness before making a paper-level RD claim.

For GLC, the analogous lesson is to keep the sparse active subset and scalar fallback, but to add a learned or decoder-safe reliability signal plus bit-aware/index-prior training before scaling active VQ/RVQ capacity.

## E185/E186 EF-LIC Deployability Lesson

A subtle but important lesson from E185 is that decoder-safe is not a single category. For a whole-image no-side-bit selector, the feature must be available before choosing the active/fallback branch. Features from later slice mean/scale are still decoder-reproducible, but only inside a sequential per-slice policy. This matters for paper claims: a selector using `slice2_mean_max` should not be described as a global predecision rule.

The stronger result is that force0 still works with stricter global predecision features. E186 directly verifies the rule `slice0_mean_abs_mean <= 0.455596` in EF-LIC's evaluation path and obtains DISTS/LPIPS gains at unchanged bpp with exact encoder/decoder decision agreement. This supports the HCG-RVQ direction but also defines the next standard: future reliability controllers should be split-trained or learned from decoder-side context, not selected and evaluated on the same Kodak table.

## 2026-06-04: EF-LIC selector control needs independent reliability, not only a strong Kodak rule

E187/E188 add an important caution to the EF-LIC plug-in track. The HCG-RVQ thesis is still supported: hyperprior/early slice context can choose when local projected geometry helps, and that choice is decoder-reproducible without side bits. However, split-fit results show that the scalar feature chosen by a tiny Kodak train split is unstable, so the paper should not overstate a hand-picked threshold.

The useful design lesson is the metric tradeoff. DISTS-only selection gives a clearer DISTS gain but can hurt LPIPS; LPIPS-target selection gives a smaller DISTS gain while also improving LPIPS. For VQ-LIC low-bitrate papers, this matters because perceptual metrics and distortion metrics do not always agree. The next EF-LIC HCG controller should therefore learn a conservative reliability score from decoder-known context and optimize a multi-metric or RD-perceptual target, while keeping the original RVQ path as exact fallback.

This aligns with the broader related-work reading: strong generative/low-bitrate codecs make careful reliability and bit accounting more important than raw residual MSE reduction. EF-LIC is still the clean direct RVQ plug-in path, and GLC remains the harder generative path where active HCG needs scalar fallback, index prior, and decoder-aware training.

## 2026-06-04: EF-LIC/GLC reproduction protocol notes from local PDFs and official repos

EF-LIC is the cleaner direct RVQ plug-in target. The official repo is inference-only and evaluates five `force_ind` rate points on a Kodak-style image folder with PSNR, LPIPS-VGG, DISTS, and BPP. The paper reports training on ImageNet and evaluates Kodak, Tecnick, DIV2K, and CLIC 2020, using LPIPS and DISTS as principal perceptual metrics and BD-rate tables. Therefore the next paper-facing EF-LIC step should be: official checkpoint reproduction, independent validation images for controller fitting, Kodak/Tecnick/DIV2K/CLIC-style evaluation when data are available, then matched fine-tuning/training if the selector signal survives.

GLC is the stronger low-bitrate generative target but a heavier integration. The paper uses progressive Stage I/II/III training, with natural-image training on ImageNet and evaluation on CLIC 2020, Kodak, DIV2K, and MS-COCO 30K. It emphasizes ultra-low bitrate (<0.04 bpp for natural images), DISTS/FID/KID, and includes PSNR/MS-SSIM as supplementary completeness metrics. Therefore GLC should keep scalar fallback and sparse active-state design, and final rows should come from Stage-II/III-compatible fine-tuning or retraining rather than only fixed codebook diagnostics.

Current workspace data are not enough for final claims: `experiments/data` contains Kodak24 and Kodak first4 only, while EF-LIC and GLC official repos here do not include full validation images. The code path is ready to evaluate arbitrary image directories, but the next independent-controller/full-eval stage needs external Kodak-like/Tecnick/DIV2K/CLIC/OpenImages images placed in the workspace or downloaded with approval.
