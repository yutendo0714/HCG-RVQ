# HCG-RVQ Research Reading Notes

Date: 2026-05-29

This note records the pre-implementation reading pass for HCG-RVQ. It is meant to preserve the project assumptions before coding starts.

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
