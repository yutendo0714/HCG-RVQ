from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
OUT_PREFIX = ROOT / "experiments" / "analysis" / "e247_loss_objective_audit"

CORE_LOSS_KEYS = {"lambda_rd", "beta_commit", "mse_scale"}
TEACHER_OR_SELECTOR_KEYS = {
    "rho_householder_reliability_teacher",
    "rho_householder_residual_selector_teacher",
    "rho_householder_residual_selector_noop",
}
ANCHOR_KEYS = {
    "rho_anchor_mu",
    "rho_anchor_log_s",
    "rho_anchor_u",
    "rho_anchor_y_hat",
    "rho_anchor_selected_distortion_margin",
}
GEOMETRY_REG_KEYS = {
    "rho_gate",
    "rho_mu_q_abs",
    "rho_s_q_std",
    "rho_householder_delta",
    "rho_householder_delta_target",
    "rho_householder_delta_local_cap",
    "rho_householder_delta_image_tail",
    "rho_householder_gate_raw_tail",
}


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def nonzero_loss_terms(loss_cfg: dict[str, Any]) -> dict[str, float]:
    terms: dict[str, float] = {}
    for key, value in loss_cfg.items():
        val = numeric(value)
        if val is None or val == 0.0:
            continue
        terms[key] = val
    return terms


def classify_terms(terms: dict[str, float]) -> tuple[list[str], list[str], list[str], list[str]]:
    core: list[str] = []
    teacher: list[str] = []
    anchor: list[str] = []
    geometry: list[str] = []
    other: list[str] = []
    for key in sorted(terms):
        if key in CORE_LOSS_KEYS:
            core.append(key)
        elif key in TEACHER_OR_SELECTOR_KEYS:
            teacher.append(key)
        elif key in ANCHOR_KEYS:
            anchor.append(key)
        elif key in GEOMETRY_REG_KEYS:
            geometry.append(key)
        else:
            other.append(key)
    return core, teacher, anchor, geometry + other


def parse_config(path: Path) -> dict[str, Any] | None:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception as exc:  # noqa: BLE001 - audit should report bad configs.
        return {"__error__": str(exc)}


def main() -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = parse_config(path)
        if not isinstance(cfg, dict):
            continue
        loss_cfg = cfg.get("loss", {}) if "__error__" not in cfg else {}
        if not isinstance(loss_cfg, dict):
            loss_cfg = {}
        terms = nonzero_loss_terms(loss_cfg)
        core, teacher, anchor, regularizers = classify_terms(terms)
        noncore = [key for key in terms if key not in CORE_LOSS_KEYS]
        rows.append(
            {
                "config": str(path.relative_to(ROOT)),
                "run_name": cfg.get("run_name", ""),
                "error": cfg.get("__error__", ""),
                "lambda_rd": terms.get("lambda_rd", ""),
                "beta_commit": terms.get("beta_commit", ""),
                "noncore_count": len(noncore),
                "teacher_selector_count": len(teacher),
                "anchor_count": len(anchor),
                "regularizer_count": len(regularizers),
                "core_terms": ",".join(core),
                "teacher_selector_terms": ",".join(teacher),
                "anchor_terms": ",".join(anchor),
                "regularizer_terms": ",".join(regularizers),
                "nonzero_terms_json": json.dumps(terms, sort_keys=True),
            }
        )

    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_PREFIX.with_suffix(".csv")
    json_path = OUT_PREFIX.with_suffix(".json")
    md_path = OUT_PREFIX.with_suffix(".md")

    fieldnames = [
        "config",
        "run_name",
        "error",
        "lambda_rd",
        "beta_commit",
        "noncore_count",
        "teacher_selector_count",
        "anchor_count",
        "regularizer_count",
        "core_terms",
        "teacher_selector_terms",
        "anchor_terms",
        "regularizer_terms",
        "nonzero_terms_json",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    total = len(rows)
    clean = [r for r in rows if int(r["noncore_count"]) == 0 and not r["error"]]
    teacher_heavy = [
        r
        for r in rows
        if int(r["teacher_selector_count"]) > 0 or int(r["anchor_count"]) > 0
    ]
    regularized = [r for r in rows if int(r["regularizer_count"]) > 0]

    def table_lines(subset: list[dict[str, Any]], limit: int = 20) -> list[str]:
        lines = [
            "| config | lambda | beta | teacher/selector | anchor | regularizers |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in subset[:limit]:
            lines.append(
                "| {config} | {lambda_rd} | {beta_commit} | {teacher_selector_count} | {anchor_count} | {regularizer_count} |".format(
                    **row
                )
            )
        return lines

    risky_sorted = sorted(
        teacher_heavy,
        key=lambda r: (
            -int(r["teacher_selector_count"]),
            -int(r["anchor_count"]),
            -int(r["regularizer_count"]),
            str(r["config"]),
        ),
    )
    regularized_sorted = sorted(
        regularized,
        key=lambda r: (-int(r["regularizer_count"]), str(r["config"])),
    )

    md_lines = [
        "# E247 Loss Objective Audit",
        "",
        "This audit scans `configs/*.yaml` and separates original codec objective terms from diagnostic teacher/selector/anchor terms.",
        "",
        f"- Configs scanned: `{total}`",
        f"- RD/commit-only configs: `{len(clean)}`",
        f"- Configs with teacher/selector or anchor losses: `{len(teacher_heavy)}`",
        f"- Configs with geometry/gate regularizers: `{len(regularized)}`",
        "",
        "## Interpretation",
        "",
        "- `lambda_rd`, `beta_commit`, and `mse_scale` are treated as the core VQ codec objective.",
        "- Teacher, selector, anchor, and strong geometry penalties are useful for diagnostics, warmup, or failure isolation, but should not define paper-main full-training claims unless explicitly ablated.",
        "- The next EF-LIC/GLC full-training candidate should keep the original R-D/perceptual objective dominant and report any auxiliary terms as weak initialization/regularization.",
        "",
        "## Teacher/Selector/Anchor Loss Configs",
        "",
        *table_lines(risky_sorted),
        "",
        "## Geometry/Gate-Regularized Configs",
        "",
        *table_lines(regularized_sorted),
    ]
    md_path.write_text("\n".join(md_lines).rstrip() + "\n")

    print(f"wrote {md_path.relative_to(ROOT)}")
    print(f"configs={total} rd_commit_only={len(clean)} teacher_or_anchor={len(teacher_heavy)} regularized={len(regularized)}")


if __name__ == "__main__":
    main()
