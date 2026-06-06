
import csv
import json
from pathlib import Path

A = Path('experiments/analysis')

def read_best(path):
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    return min(rows, key=lambda row: float(row['rd_score']))

def read_first(path):
    with Path(path).open(newline='') as f:
        return next(csv.DictReader(f))

def jf(path):
    return json.loads(Path(path).read_text())

def fl(row, key):
    return float(row[key])

def fmt(x, sign=False):
    return ('%+.6f' if sign else '%.6f') % float(x)

trusted = {
    'HCS May29 trusted': read_best(A / 'pilot_hcs_rvq_frozen_seed1234_openimages_val4096_holdout4096_current.csv'),
    'old gate0.25 May29 trusted': read_best(A / 'pilot_hcg_rvq_h_gate025_seed1234_openimages_val4096_holdout4096_current.csv'),
    'min090 May29 trusted': read_best(A / 'pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_seed1234_openimages_val4096_holdout4096_current.csv'),
}
current = {
    'HCS current recheck': read_first(A / 'pilot_hcs_rvq_frozen_seed1234_openimages_val4096_holdout4096_current_recheck_after_localstats.csv'),
    'old gate0.25 current recheck': read_first(A / 'pilot_hcg_rvq_h_gate025_seed1234_openimages_val4096_holdout4096_current_recheck_after_localstats.csv'),
    'min090 current recheck': read_first(A / 'pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_seed1234_openimages_val4096_holdout4096_current_recheck_after_localstats.csv'),
    'local cap080 rho1 current': read_first(A / 'pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_seed1234_openimages_val4096_holdout4096_current.csv'),
    'local band063 cap080 rho5+1 current': read_first(A / 'pilot_hcg_rvq_h_gate025_risk_inv_detach_s044_min090_local_delta_band063_cap080_rho5_1_seed1234_openimages_val4096_holdout4096_current.csv'),
}
trusted_hcs = fl(trusted['HCS May29 trusted'], 'rd_score')
current_hcs = fl(current['HCS current recheck'], 'rd_score')
feature_files = {
    'old gate0.25 current localstats': A / 'per_image_features_hcg_h_gate025_seed1234_step250_val4096_holdout4096_current_localstats.json',
    'local cap080 rho1 current': A / 'per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_local_delta_cap080_rho1_seed1234_step250_val4096_holdout4096_current.json',
    'local band063 cap080 rho5+1 current': A / 'per_image_features_hcg_h_gate025_risk_inv_detach_s044_min090_local_delta_band063_cap080_rho5_1_seed1234_step250_val4096_holdout4096_current.json',
}
feature_keys = [('s_q','s_q_mean_mean'),('raw_gate','householder_gate_raw_mean_mean'),('risk','householder_risk_multiplier_mean_mean'),('strength','householder_strength_mean_mean'),('delta_rms','householder_delta_rms_mean'),('local_mean','rvq_householder_delta_rms_local_mean_mean'),('local_max','rvq_householder_delta_rms_local_max_mean'),('local_std','rvq_householder_delta_rms_local_std_mean'),('latent_qmse','rvq_latent_quant_mse_mean'),('index_bpp','index_empirical_bpp_mean'),('dead','rvq_dead_code_ratio_mean')]
features = {}
for name, path in feature_files.items():
    d = jf(path)
    features[name] = {out: d.get(key) for out, key in feature_keys}
summary = {'trusted_may29': trusted, 'current_recheck': current, 'current_deltas_vs_hcs': {k: fl(v, 'rd_score') - current_hcs for k, v in current.items()}, 'trusted_to_current_shift': {'HCS': current_hcs - trusted_hcs, 'old_gate025': fl(current['old gate0.25 current recheck'], 'rd_score') - fl(trusted['old gate0.25 May29 trusted'], 'rd_score'), 'min090': fl(current['min090 current recheck'], 'rd_score') - fl(trusted['min090 May29 trusted'], 'rd_score')}, 'features': features}
json_path = A / 'local_delta_controls_current_code_consistency_seed1234_holdout4096.json'
json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
lines = ['# Local Delta Controls Current-Code Consistency Audit', '', 'Seed1234 OpenImages holdout4096 audit after adding local Householder delta diagnostics. This separates stale May29 trusted CSVs from May30 current-code rechecks.', '', '## Checkpoint RD', '', '| method | protocol | checkpoint | RD | delta vs protocol HCS | bpp | PSNR | MS-SSIM |', '|---|---|---:|---:|---:|---:|---:|---:|']
for name, row in trusted.items():
    method = name.replace(' May29 trusted', '')
    delta = fl(row, 'rd_score') - trusted_hcs
    lines.append('| %s | May29 trusted CSV | %s | %s | %s | %s | %s | %s |' % (method, row['step'], fmt(fl(row, 'rd_score')), fmt(delta, True), fmt(fl(row, 'bpp')), fmt(fl(row, 'psnr')), fmt(fl(row, 'ms_ssim'))))
for name, row in current.items():
    method = name.replace(' current recheck', '').replace(' current', '')
    delta = fl(row, 'rd_score') - current_hcs
    lines.append('| %s | May30 current-code recheck | %s | %s | %s | %s | %s | %s |' % (method, row['step'], fmt(fl(row, 'rd_score')), fmt(delta, True), fmt(fl(row, 'bpp')), fmt(fl(row, 'psnr')), fmt(fl(row, 'ms_ssim'))))
lines += ['', '## Current-Code Intermediate Features', '', '| method | s_q | raw gate | risk | strength | delta RMS | local mean | local max | local std | latent qMSE | index bpp | dead code |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for name, fs in features.items():
    method = name.replace(' current localstats', '').replace(' current', '')
    values = []
    for out, key in feature_keys:
        values.append('n/a' if fs[out] is None else fmt(float(fs[out])))
    lines.append('| %s | %s |' % (method, ' | '.join(values)))
lines += ['', '## Interpretation', '', '- The May29 trusted baseline CSVs are not reproducible under the current code state: HCS shifts from 2.211475 to 2.889062 RD, old gate0.25 from 2.193095 to 2.843577, and min090 from 2.228412 to 2.891932.', '- Mixed comparisons between May29 trusted baselines and newly trained May30 local-control checkpoints are not paper-safe.', '- Within the current-code recheck only, local cap080/rho1 is the best seed1234 checkpoint among these rows: 2.828428 RD, which is -0.060633 vs current HCS and -0.015149 vs current old gate0.25.', '- The mechanism remains delicate. local cap improves current-code RD while suppressing geometry strength/delta; band restores global delta RMS near old but still fails because effective strength drops, latent qMSE doubles, and dead-code ratio rises.', '- Immediate next action: pin or restore the evaluation/model code that produced the May29 trusted CSVs, then rerun all baselines and local-control variants in one protocol before making paper-facing claims.']
md_path = A / 'local_delta_controls_current_code_consistency_seed1234_holdout4096.md'
md_path.write_text('\n'.join(lines) + '\n')
print(md_path)
print(json_path)
