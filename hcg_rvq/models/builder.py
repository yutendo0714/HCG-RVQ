from __future__ import annotations

from .hyperprior_rvq import HCGMeanScaleHyperprior


def build_model(config: dict) -> HCGMeanScaleHyperprior:
    model_cfg = config.get("model", {})
    quant_cfg = config.get("quantizer", {})
    entropy_cfg = config.get("entropy_index", {})
    gate_cfg = config.get("stage_gate", {})

    return HCGMeanScaleHyperprior(
        N=model_cfg.get("N", 192),
        M=model_cfg.get("M", 320),
        variant=quant_cfg.get("type", "scalar"),
        group_size=quant_cfg.get("group_size", 32),
        num_stages=quant_cfg.get("num_stages", 2),
        codebook_size=quant_cfg.get("codebook_size", 256),
        index_prior_enabled=entropy_cfg.get("enabled", False),
        index_hidden_channels=entropy_cfg.get("hidden_channels", 192),
        stage_gate_enabled=gate_cfg.get("enabled", False),
        scale_min=config.get("hyper_conditioning", {}).get("scale_min", 0.05),
        scale_max=config.get("hyper_conditioning", {}).get("scale_max", 10.0),
    )

