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
        use_global_norm=quant_cfg.get("use_global_norm", False),
        codebook_init_scale=quant_cfg.get("codebook_init_scale", 0.02),
        scale_min=config.get("hyper_conditioning", {}).get("scale_min", 0.05),
        scale_max=config.get("hyper_conditioning", {}).get("scale_max", 10.0),
        householder_strength=quant_cfg.get("householder_strength", 1.0),
        householder_bias_init_scale=quant_cfg.get("householder_bias_init_scale", 0.0),
        householder_gate_enabled=quant_cfg.get("householder_gate_enabled", False),
        householder_gate_max=quant_cfg.get("householder_gate_max", 0.45),
        householder_gate_init=quant_cfg.get("householder_gate_init", 0.25),
        householder_gate_risk_enabled=quant_cfg.get("householder_gate_risk_enabled", False),
        householder_gate_risk_center=quant_cfg.get("householder_gate_risk_center", 0.56),
        householder_gate_risk_sharpness=quant_cfg.get("householder_gate_risk_sharpness", 12.0),
        householder_gate_risk_min=quant_cfg.get("householder_gate_risk_min", 0.5),
        householder_gate_risk_invert=quant_cfg.get("householder_gate_risk_invert", False),
        householder_gate_risk_detach=quant_cfg.get("householder_gate_risk_detach", False),
        householder_gate_reliability_enabled=quant_cfg.get("householder_gate_reliability_enabled", False),
        householder_gate_reliability_min=quant_cfg.get("householder_gate_reliability_min", 0.5),
        householder_gate_reliability_init=quant_cfg.get("householder_gate_reliability_init", 0.99),
        householder_gate_reliability_detach=quant_cfg.get("householder_gate_reliability_detach", False),
        householder_gate_raw_backoff_enabled=quant_cfg.get("householder_gate_raw_backoff_enabled", False),
        householder_gate_raw_backoff_threshold=quant_cfg.get("householder_gate_raw_backoff_threshold", 0.284059),
        householder_gate_raw_backoff_min=quant_cfg.get("householder_gate_raw_backoff_min", 0.65),
        householder_gate_raw_backoff_sharpness=quant_cfg.get("householder_gate_raw_backoff_sharpness", 80.0),
        householder_gate_raw_backoff_detach=quant_cfg.get("householder_gate_raw_backoff_detach", True),
        householder_gate_raw_backoff_use_image_mean=quant_cfg.get("householder_gate_raw_backoff_use_image_mean", True),
        householder_gate_strength_backoff_enabled=quant_cfg.get("householder_gate_strength_backoff_enabled", False),
        householder_gate_strength_backoff_threshold=quant_cfg.get(
            "householder_gate_strength_backoff_threshold", 0.271352783
        ),
        householder_gate_strength_backoff_min=quant_cfg.get("householder_gate_strength_backoff_min", 0.0),
        householder_gate_strength_backoff_sharpness=quant_cfg.get(
            "householder_gate_strength_backoff_sharpness", 80.0
        ),
        householder_gate_strength_backoff_detach=quant_cfg.get("householder_gate_strength_backoff_detach", True),
        householder_gate_strength_backoff_use_image_mean=quant_cfg.get(
            "householder_gate_strength_backoff_use_image_mean", True
        ),
        householder_gate_residual_selector_enabled=quant_cfg.get("householder_gate_residual_selector_enabled", False),
        householder_gate_residual_selector_max=quant_cfg.get("householder_gate_residual_selector_max", 0.50),
        householder_gate_residual_selector_bias=quant_cfg.get("householder_gate_residual_selector_bias", -4.0),
        householder_gate_residual_selector_deadzone_threshold=quant_cfg.get(
            "householder_gate_residual_selector_deadzone_threshold", 0.0
        ),
        householder_gate_residual_selector_detach=quant_cfg.get("householder_gate_residual_selector_detach", True),
    )

