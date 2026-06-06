import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hcg_rvq.utils import load_config


SEEDS = (1234, 2345, 3456)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def cfg_path(kind: str, seed: int) -> Path:
    if kind == "old":
        return Path(f"configs/pilot_hcg_rvq_h_gate025_frozen_seed{seed}.yaml")
    if kind == "risk":
        return Path(f"configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_frozen_seed{seed}.yaml")
    raise ValueError(kind)


def selected_fields(config: dict[str, Any]) -> dict[str, Any]:
    quantizer = config.get("quantizer", {})
    train = config.get("train", {})
    return {
        "run_name": config.get("run_name"),
        "quantizer_type": quantizer.get("type"),
        "group_size": quantizer.get("group_size"),
        "num_stages": quantizer.get("num_stages"),
        "codebook_size": quantizer.get("codebook_size"),
        "householder_gate_enabled": quantizer.get("householder_gate_enabled", False),
        "householder_gate_init": quantizer.get("householder_gate_init"),
        "householder_gate_risk_enabled": quantizer.get("householder_gate_risk_enabled", False),
        "householder_gate_risk_center": quantizer.get("householder_gate_risk_center"),
        "householder_gate_risk_min": quantizer.get("householder_gate_risk_min"),
        "householder_gate_risk_invert": quantizer.get("householder_gate_risk_invert", False),
        "householder_gate_risk_detach": quantizer.get("householder_gate_risk_detach", False),
        "init_model": train.get("init_model"),
        "freeze_prefixes": train.get("freeze_prefixes", []),
    }


def config_audit() -> dict[str, Any]:
    rows = []
    for seed in SEEDS:
        old_cfg = selected_fields(load_config(cfg_path("old", seed)))
        risk_cfg = selected_fields(load_config(cfg_path("risk", seed)))
        rows.append(
            {
                "seed": seed,
                "old": old_cfg,
                "risk": risk_cfg,
                "same_run_name": old_cfg["run_name"] == risk_cfg["run_name"],
                "same_arch_family": all(
                    old_cfg[key] == risk_cfg[key]
                    for key in ("quantizer_type", "group_size", "num_stages", "codebook_size", "householder_gate_enabled")
                ),
                "same_gate_policy": all(
                    old_cfg[key] == risk_cfg[key]
                    for key in (
                        "householder_gate_risk_enabled",
                        "householder_gate_risk_center",
                        "householder_gate_risk_min",
                        "householder_gate_risk_invert",
                        "householder_gate_risk_detach",
                    )
                ),
                "same_init_model": old_cfg["init_model"] == risk_cfg["init_model"],
            }
        )
    return {
        "rows": rows,
        "all_same_run_name": all(row["same_run_name"] for row in rows),
        "all_same_arch_family": all(row["same_arch_family"] for row in rows),
        "all_same_gate_policy": all(row["same_gate_policy"] for row in rows),
        "all_same_init_model": all(row["same_init_model"] for row in rows),
    }


def split_row(report: dict[str, Any], split: str) -> dict[str, Any]:
    data = report["splits"][split]
    selected = data["selected_policy"]
    baseline = data["baseline"]
    return {
        "hcs_rd": data["hcs_mean_rd"],
        "old_rd": data["old_mean_rd"],
        "risk_rd": data["risk_mean_rd"],
        "oracle_rd": data["oracle_old_or_min090_mean_rd"],
        "selected_rd": selected["mean_rd"],
        "old_delta": baseline["old_gate_mean_delta"],
        "risk_delta": baseline["min090_mean_delta"],
        "oracle_delta": baseline["oracle_old_or_min090_delta"],
        "selected_delta": selected["mean_delta"],
        "selected_vs_old": selected["vs_old_gate025"],
        "selected_oracle_gap_closed": selected["oracle_gap_closed_fraction"],
        "selected_risk_fraction": selected["risk_fraction"],
    }


def fmt(value: float, signed: bool = False) -> str:
    return ("{:+.6f}" if signed else "{:.6f}").format(value)


def make_payload(analysis_dir: Path) -> dict[str, Any]:
    report = read_json(analysis_dir / "gate025_min090_selector_reporting_protocol.json")
    cfg = config_audit()
    splits = {
        "validation_holdout4096": split_row(report, "validation_holdout4096"),
        "reporting_start0_transfer": split_row(report, "reporting_start0_transfer"),
        "reporting_start0_slice_best": split_row(report, "reporting_start0_slice_best"),
    }
    pixels_256 = 256 * 256
    return {
        "calibrated_policy": report["calibrated_policy"],
        "config_audit": cfg,
        "splits": splits,
        "claim_tiers": {
            "single_model_old_gate025": {
                "status": "paper_safe_baseline_or_main_single_checkpoint_variant",
                "why": "old gate0.25 is one trained checkpoint/config per seed and does not require switching models per image.",
            },
            "single_model_min090": {
                "status": "valid_single_checkpoint_diagnostic_not_current_main",
                "why": "min090 risk is deterministic inside one model, but transfer/slice-best results are weaker than old gate0.25.",
            },
            "calibrated_old_min090_selector": {
                "status": "evidence_for_reliability_control_not_yet_single_codec_main_claim",
                "why": "the selector currently chooses between separately trained old and min090 checkpoints, even though the selection feature itself is decoder-reproducible.",
            },
        },
        "model_switch_overhead_bpp_for_one_flag_per_256x256_image": 1.0 / pixels_256,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    policy = payload["calibrated_policy"]
    cfg = payload["config_audit"]
    lines = [
        "# Selector Claim-Readiness Audit",
        "",
        "This audit separates numerical selector evidence from what can be claimed as a single-codec method.",
        "",
        "## Calibrated Selector Evidence",
        "",
        f"- policy: `{policy['feature']} {policy['direction']} {float(policy['threshold']):.6f}`",
        "- selection feature is decoder-side in principle because it is derived from hyperprior/Householder gate statistics.",
        "- current evaluated selector still switches between separately trained old gate0.25 and min090 checkpoints.",
        "",
        "| split | old delta | min090 delta | selected delta | selected vs old | oracle gap closed | min090 fraction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload["splits"].items():
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                name,
                fmt(row["old_delta"], True),
                fmt(row["risk_delta"], True),
                fmt(row["selected_delta"], True),
                fmt(row["selected_vs_old"], True),
                fmt(row["selected_oracle_gap_closed"]),
                fmt(row["selected_risk_fraction"]),
            )
        )

    lines.extend(
        [
            "",
            "## Config/Checkpoint Readiness",
            "",
            "| property | result | implication |",
            "| --- | --- | --- |",
            f"| same architecture family | `{cfg['all_same_arch_family']}` | old/min090 are comparable HCG-RVQ-H variants. |",
            f"| same gate policy | `{cfg['all_same_gate_policy']}` | `False` means min090 is a different deterministic gate rule, not just a reporting-time threshold. |",
            f"| same run/checkpoint | `{cfg['all_same_run_name']}` | `False` means the current per-image selector is a multi-checkpoint/codec-selection diagnostic. |",
            f"| same initialization | `{cfg['all_same_init_model']}` | shared initialization reduces confounding but does not make the checkpoints identical. |",
            "",
            "## Claim Tiers",
            "",
            "| candidate | status | paper use |",
            "| --- | --- | --- |",
            "| old gate0.25 | single-checkpoint variant | Safe as the current main single-model HCG geometry result. |",
            "| min090 risk | single-checkpoint diagnostic | Useful ablation; not current main because transfer/slice-best behavior is weaker. |",
            "| calibrated old/min090 selector | multi-checkpoint diagnostic | Strong evidence that reliability control has headroom; do not present as final single-codec method yet. |",
            "",
            "## Practical Note",
            "",
            f"A one-bit image-level model flag would cost about `{payload['model_switch_overhead_bpp_for_one_flag_per_256x256_image']:.8f}` bpp for a 256x256 crop, so signaling overhead is tiny. The real issue is not the bit cost; it is that a model-switching ensemble is a different codec protocol and a weaker paper claim than a unified HCG-RVQ checkpoint.",
            "",
            "## Next Method Action",
            "",
            "Promote the selector into a single-checkpoint reliability controller before making it the main method claim. The cleanest next target is a unified gate policy trained/evaluated in one model, then compare it against HCS, old gate0.25, min090 risk, and the multi-checkpoint selector headroom.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="experiments/analysis")
    parser.add_argument("--output-prefix", default="selector_claim_readiness")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    payload = make_payload(analysis_dir)
    json_path = analysis_dir / f"{args.output_prefix}.json"
    md_path = analysis_dir / f"{args.output_prefix}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(payload))
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
