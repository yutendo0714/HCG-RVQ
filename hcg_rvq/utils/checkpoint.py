from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"unsupported checkpoint type: {type(checkpoint)!r}")


def load_matching_state_dict(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    skip_prefixes: tuple[str, ...] = (),
) -> dict[str, int]:
    """Load only parameters whose names and shapes match the target model."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    source = extract_state_dict(checkpoint)
    target = model.state_dict()

    loaded: dict[str, torch.Tensor] = {}
    skipped_prefix = 0
    skipped_shape = 0
    skipped_missing = 0

    for name, tensor in source.items():
        if any(name.startswith(prefix) for prefix in skip_prefixes):
            skipped_prefix += 1
            continue
        if name not in target:
            skipped_missing += 1
            continue
        if target[name].shape != tensor.shape:
            skipped_shape += 1
            continue
        loaded[name] = tensor

    target.update(loaded)
    model.load_state_dict(target)
    return {
        "loaded": len(loaded),
        "skipped_prefix": skipped_prefix,
        "skipped_shape": skipped_shape,
        "skipped_missing": skipped_missing,
        "target_total": len(target),
    }
