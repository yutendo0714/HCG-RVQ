
import csv
import json
import math
from pathlib import Path

A = Path("experiments/analysis")
old_comp = A / "per_image_seed1234_hcs500_vs_hcgh_gate025_step250_val4096_holdout4096_current.csv"
min_comp = A / "per_image_seed1234_hcs500_vs_hcgh_gate025_risk_inv_detach_s044_min090_step500_val4096_holdout4096_current.csv"
old_feat = A / "per_image_features_hcg_h_gate025_seed1234_step250_val4096_holdout4096_current.csv"
min_feat = A / "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_seed1234_step500_val4096_holdout4096_current.csv"
loc_feat = A / "per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_local_delta_band063_cap080_rho5_1_seed1234_step250_val4096_holdout4096_current.csv"

def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))

def f(value):
    return float(value)

def mean(values):
    values = list(values)
    return sum(values) / len(values)

def corr(xs, ys):
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)

def fmt(value, sign=False):
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{value:+.6f}" if sign else f"{value:.6f}"

old_rows = rows(old_comp)
min_rows = rows(min_comp)
loc_rows = rows(loc_feat)
min_by_path = {row["path"]: row for row in min_rows}
loc_by_path = {row["path"]: row for row in loc_rows}
joined = []
for row in old_rows:
    path = row["path"]
    min_row = min_by_path[path]
    loc_row = loc_by_path[path]
    item = {
        "index": int(row["index"]),
        "path": path,
        "hcs_rd": f(row["HCS_rd_score"]),
        "old_rd": f(row["old_gate025_rd_score"]),
        "min090_rd": f(min_row["min090_rd_score"]),
        "local_rd": f(loc_row["rd_score"]),
        "local_s_q_mean": f(loc_row["s_q_mean"]),
        "local_strength": f(loc_row["householder_strength_mean"]),
        "local_delta_rms": f(loc_row["householder_delta_rms"]),
        "local_qmse": f(loc_row["rvq_latent_quant_mse"]),
        "local_risk": f(loc_row["householder_risk_multiplier_mean"]),
        "local_dead": f(loc_row["rvq_dead_code_ratio"]),
        "local_index_bpp": f(loc_row["index_empirical_bpp"]),
    }
    item['old_delta_vs_hcs'] = item["old_rd"] - item['hcs_rd']
    item['min090_delta_vs_hcs'] = item["min090_rd"] - item['hcs_rd']
    item['local_delta_vs_hcs'] = item["local_rd"] - item['hcs_rd']
    item['local_delta_vs_old'] = item["local_rd"] - item["old_rd"]
    item['local_delta_vs_min090'] = item["local_rd"] - item["min090_rd"]
    joined.append(item)

def method_stats(name, field):
    return {
        "name": name,
        "rd": mean(row[field] for row in joined),
        "delta_vs_hcs": mean(row[field] - row['hcs_rd'] for row in joined),
        "win_vs_hcs": sum(1 for row in joined if row[field] < row['hcs_rd']) / len(joined),
    }

methods = {
    "HCS": {"rd": mean(row['hcs_rd'] for row in joined)},
    "old gate0.25": method_stats("old gate0.25", "old_rd"),
    "trained min090 risk": method_stats("trained min090 risk", "min090_rd"),
    "local band063 cap080 rho5+1": method_stats("local band063 cap080 rho5+1", "local_rd"),
}
for name in ["old gate0.25", "trained min090 risk", "local band063 cap080 rho5+1"]:
    methods[name]['delta_vs_old'] = methods[name]['rd'] - methods["old gate0.25"]['rd']
    methods[name]['delta_vs_min090'] = methods[name]['rd'] - methods["trained min090 risk"]['rd']

def feature_summary(path):
    source_rows = rows(path)
    keys = [
        ("s_q", "s_q_mean"),
        ("raw_gate", "householder_gate_raw_mean"),
        ("risk_multiplier", "householder_risk_multiplier_mean"),
        ("strength", "householder_strength_mean"),
        ("delta_rms", "householder_delta_rms"),
        ("latent_qmse", "rvq_latent_quant_mse"),
        ("index_bpp", "index_empirical_bpp"),
        ("dead_code", "rvq_dead_code_ratio"),
    ]
    out = {}
    for out_key, key in keys:
        vals = [f(row[key]) for row in source_rows if key in row and row[key] != ""]
        out[out_key] = mean(vals) if vals else None
    return out

features = {
    "old gate0.25": feature_summary(old_feat),
    "trained min090 risk": feature_summary(min_feat),
    "local band063 cap080 rho5+1": feature_summary(loc_feat),
}

sorted_rows = sorted(joined, key=lambda row: row['hcs_rd'])
quartiles = []
n = len(sorted_rows)
for qi in range(4):
    chunk = sorted_rows[qi * n // 4 : (qi + 1) * n // 4]
    quartiles.append({
        "quartile": qi + 1,
        "n": len(chunk),
        "hcs_rd": mean(row['hcs_rd'] for row in chunk),
        "old_delta_vs_hcs": mean(row['old_delta_vs_hcs'] for row in chunk),
        "min090_delta_vs_hcs": mean(row['min090_delta_vs_hcs'] for row in chunk),
        "local_delta_vs_hcs": mean(row['local_delta_vs_hcs'] for row in chunk),
        "local_delta_vs_old": mean(row['local_delta_vs_old'] for row in chunk),
        "local_delta_vs_min090": mean(row['local_delta_vs_min090'] for row in chunk),
        "local_delta_rms": mean(row['local_delta_rms'] for row in chunk),
        "local_strength": mean(row['local_strength'] for row in chunk),
        "local_qmse": mean(row['local_qmse'] for row in chunk),
        "local_dead": mean(row['local_dead'] for row in chunk),
    })

correlations = {
    key: corr([row['local_delta_vs_hcs'] for row in joined], [row[key] for row in joined])
    for key in ["local_s_q_mean", "local_strength", "local_delta_rms", "local_qmse", "local_risk", "local_dead", "local_index_bpp", "hcs_rd"]
}
summary = {
    "methods": methods,
    "features": features,
    "quartiles": quartiles,
    "correlations_with_local_delta_vs_hcs": correlations,
    "artifacts": {
        "config": "configs/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_band063_cap080_rho5_1_frozen_seed1234.yaml",
        "checkpoint": "experiments/pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_band063_cap080_rho5_1_frozen_g64_l1_k128_lambda0035_seed1234/checkpoint_step_250.pth.tar",
        "feature_distribution": "experiments/analysis/feature_distribution_hcg_h_gate025_risk_inv_detach_s044_min090_local_delta_band063_cap080_rho5_1_seed1234_step250_val4096_holdout4096_current.json",
        "per_image_features": str(loc_feat),
    },
    "num_images": len(joined),
}
json_path = A / "local_delta_band063_cap080_rho5_1_seed1234_val4096_holdout4096_current.json"
json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

lines = [
    "# Local Delta Band Probe Audit",
    "",
    "Seed1234 holdout4096 audit for `rho_householder_delta_target=5.0`, `householder_delta_target=0.063`, `rho_householder_delta_local_cap=1.0`, and `householder_delta_local_cap=0.080` on the min090 inverse/detached risk gate.",
    "",
    "## Overall RD",
    "",
    "| method | mean RD | delta vs HCS | delta vs old | delta vs min090 | win vs HCS |",
    "|---|---:|---:|---:|---:|---:|",
    f"| HCS | {fmt(methods['HCS']['rd'])} | {fmt(0, True)} | n/a | n/a | n/a |",
]
for name in ["old gate0.25", "trained min090 risk", "local band063 cap080 rho5+1"]:
    st = methods[name]
    lines.append(f"| {name} | {fmt(st['rd'])} | {fmt(st['delta_vs_hcs'], True)} | {fmt(st['delta_vs_old'], True)} | {fmt(st['delta_vs_min090'], True)} | {fmt(st['win_vs_hcs'])} |")
lines += [
    "",
    "## Intermediate Features",
    "",
    "| method | s_q | raw gate | risk mult | strength | delta RMS | latent qMSE | index bpp | dead code |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for name in ["old gate0.25", "trained min090 risk", "local band063 cap080 rho5+1"]:
    fs = features[name]
    lines.append(f"| {name} | {fmt(fs['s_q'])} | {fmt(fs['raw_gate'])} | {fmt(fs['risk_multiplier'])} | {fmt(fs['strength'])} | {fmt(fs['delta_rms'])} | {fmt(fs['latent_qmse'])} | {fmt(fs['index_bpp'])} | {fmt(fs['dead_code'])} |")
lines += [
    "",
    "## HCS-Difficulty Quartiles",
    "",
    "| Q | HCS RD | old-HCS | min090-HCS | local-HCS | local-old | local-min090 | local delta RMS | local strength | local qMSE | local dead |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in quartiles:
    lines.append(f"| {row['quartile']} | {fmt(row['hcs_rd'])} | {fmt(row['old_delta_vs_hcs'], True)} | {fmt(row['min090_delta_vs_hcs'], True)} | {fmt(row['local_delta_vs_hcs'], True)} | {fmt(row['local_delta_vs_old'], True)} | {fmt(row['local_delta_vs_min090'], True)} | {fmt(row['local_delta_rms'])} | {fmt(row['local_strength'])} | {fmt(row['local_qmse'])} | {fmt(row['local_dead'])} |")
lines += [
    "",
    "## Correlations",
    "",
    "| feature | corr with local-HCS RD delta |",
    "|---|---:|",
]
for key, value in correlations.items():
    lines.append(f"| {key} | {fmt(value, True)} |")
lines += [
    "",
    "## Interpretation",
    "",
    "- The local band is rejected as configured. Matching a scalar global delta target while capping local peaks does not recover the decoder-compatible quantization regime.",
    "- The best checkpoint has much higher RD than HCS/old/min090. The global delta RMS lands near old gate0.25, but effective strength is too low, local delta peaks are too high, latent qMSE is high, and dead-code ratio rises sharply.",
    "- This strengthens the E050/E051 diagnosis: scalar geometry magnitude is not the causal variable by itself. The next controller must preserve a joint operating regime: scale, effective strength, local-delta tails, latent qMSE, and codebook usage.",
]
md_path = A / "local_delta_band063_cap080_rho5_1_seed1234_val4096_holdout4096_current.md"
md_path.write_text("\n".join(lines) + "\n")
print(md_path)
print(json_path)
print("local RD", methods["local band063 cap080 rho5+1"]['rd'])
print("local-HCS", methods["local band063 cap080 rho5+1"]['delta_vs_hcs'])
