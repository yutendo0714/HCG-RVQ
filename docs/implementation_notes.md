# Implementation Notes

## Model Rate Accounting

For scalar baselines, y-rate comes from `GaussianConditional` likelihoods and z-rate comes from `EntropyBottleneck`.

For RVQ variants, y-rate must come from RVQ index coding:

- fixed-length proxy when no index entropy model is enabled;
- categorical cross-entropy proxy when the hyper index prior is enabled;
- eventual range-coded bits for final bitstream experiments.

Do not add Gaussian y likelihood and RVQ index rate at the same time.

## Initial RVQ Defaults

The first pilots use conservative settings:

```yaml
group_size: 32
num_stages: 2
codebook_size: 256
```

The larger `group_size=16, num_stages=4` setup is left for later because its symbol budget is aggressive at low bpp.

## Stage Gate Warning

Soft gate values do not reduce bitrate unless skipped stages are not transmitted. Later gate experiments should use a deterministic or annealed hard skip mask derived from `z_hat`.

