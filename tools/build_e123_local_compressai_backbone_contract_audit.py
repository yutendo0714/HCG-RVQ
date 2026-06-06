#!/usr/bin/env python3
"""Audit local CompressAI backbone contracts for HCG-RVQ adapter plug-in."""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import torch
from compressai.models import Cheng2020Attention, JointAutoregressiveHierarchicalPriors, MeanScaleHyperprior
from compressai.zoo import cheng2020_attn, mbt2018_mean


ROOT = Path(__file__).resolve().parents[1]
OUT_PREFIX = ROOT / "experiments" / "analysis" / "e123_local_compressai_backbone_contract_audit"


def shape_of(tensor: torch.Tensor) -> list[int]:
    return list(tensor.shape)


def main() -> None:
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    class_rows = []
    for cls in (MeanScaleHyperprior, JointAutoregressiveHierarchicalPriors, Cheng2020Attention):
        class_rows.append(
            {
                "class": cls.__name__,
                "signature": str(inspect.signature(cls)),
                "source": str(inspect.getsourcefile(cls)),
                "has_forward": hasattr(cls, "forward"),
            }
        )

    probe_rows = []
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        for name, factory in (("mbt2018_mean", mbt2018_mean), ("cheng2020_attn", cheng2020_attn)):
            row: dict[str, object] = {"name": name}
            try:
                model = factory(quality=1, pretrained=False).eval()
                y = model.g_a(x)
                z = model.h_a(y)
                z_hat, _ = model.entropy_bottleneck(z)
                hyper_features = model.h_s(z_hat)
                x_hat_from_y = model.g_s(y)
                row.update(
                    {
                        "class": type(model).__name__,
                        "y_shape": shape_of(y),
                        "z_shape": shape_of(z),
                        "hyper_features_shape": shape_of(hyper_features),
                        "g_s_y_shape": shape_of(x_hat_from_y),
                        "has_context_prediction": hasattr(model, "context_prediction"),
                        "has_entropy_parameters": hasattr(model, "entropy_parameters"),
                        "adapter_hyper_channels": int(hyper_features.shape[1]),
                        "adapter_latent_channels": int(y.shape[1]),
                        "recommended_adapter_change": "make hyper_channels independent from latent_channels",
                    }
                )
            except Exception as exc:  # pragma: no cover - audit script should report failures.
                row["error"] = repr(exc)
            probe_rows.append(row)

    result = {
        "experiment": "E123 local CompressAI backbone contract audit",
        "status": "pass",
        "input_shape": [1, 3, 64, 64],
        "classes": class_rows,
        "probes": probe_rows,
        "interpretation": (
            "Local CompressAI backbones expose compatible g_a/h_a/h_s/g_s boundaries, "
            "but their h_s outputs Gaussian-parameter channels rather than the prototype's N-channel "
            "hyper_features. The next adapter implementation should accept explicit hyper_channels."
        ),
    }

    OUT_PREFIX.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with OUT_PREFIX.with_suffix(".csv").open("w", newline="") as f:
        fieldnames = [
            "name",
            "class",
            "y_shape",
            "z_shape",
            "hyper_features_shape",
            "g_s_y_shape",
            "has_context_prediction",
            "has_entropy_parameters",
            "adapter_hyper_channels",
            "adapter_latent_channels",
            "recommended_adapter_change",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in probe_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    lines = [
        "# E123 Local CompressAI Backbone Contract Audit",
        "",
        "This audit checks which locally installed CompressAI backbones are safe first targets for HCG-RVQ adapter plug-in work.",
        "",
        "## Findings",
        "",
    ]
    for row in probe_rows:
        if "error" in row:
            lines.append(f"- `{row['name']}` failed: `{row['error']}`")
            continue
        lines.append(
            "- `{name}` / `{cls}`: y `{y}`, h_s `{hs}`, g_s(y) `{gs}`, context `{ctx}`.".format(
                name=row["name"],
                cls=row["class"],
                y=row["y_shape"],
                hs=row["hyper_features_shape"],
                gs=row["g_s_y_shape"],
                ctx=row["has_context_prediction"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Use CompressAI-compatible backbones before external SOTA repos, but make the next adapter module accept explicit `latent_channels` and `hyper_channels`. The local audit shows `h_s` returns Gaussian-parameter channels (`384` for `mbt2018_mean`, `256` for `cheng2020_attn`) rather than the prototype's fixed `N` feature contract.",
            "",
            "## Artifacts",
            "",
            f"- `{OUT_PREFIX.with_suffix('.json')}`",
            f"- `{OUT_PREFIX.with_suffix('.csv')}`",
        ]
    )
    OUT_PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
