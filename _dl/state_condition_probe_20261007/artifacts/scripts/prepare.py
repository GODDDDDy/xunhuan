"""Lock existing evidence and the next diagnostic before any new forwards."""
import copy,hashlib,json,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parent.parent
R=ROOT.parent
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
prior=read(R/'test_completion_20261007/protocol_lock.json')
assert read(R/'test_completion_20261007/status.json')['state']=='complete'
assert all(sha(p)==h for p,h in prior['locked_file_sha256'].items())
u=next(u for u in prior['units'] if u['id']=='K2_aligned_seed2026')
diag_cfg=yaml.safe_load(Path(u['source_config']).read_text())
diag_cfg['paths']['output_root']=str(ROOT/'diagnostics/current_K2_build')
(ROOT/'configs/diagnostic.yaml').write_text(yaml.safe_dump(diag_cfg,sort_keys=False))
proxy=read(diag_cfg['paths']['validation_proxy'])
assert proxy['sample_count']==28 and len(proxy['sample_ids'])==28
assert all(x.startswith('validation/') for x in proxy['sample_ids'])
f=R/'frozen_g6_adapter_20261007'
source_cfg=f/'configs/K1_seed2026.yaml'
for arm in ('mixed_rgb','task_prediction'):
    cfg=yaml.safe_load(source_cfg.read_text())
    cfg['paths']['output_root']=str(ROOT/'runs'/arm/'seed_2026')
    cfg['optimization']['total_steps']=1000
    cfg['optimization']['validation_steps']=[0,200,1000]
    cfg['study'].update(fixed_budget=1000,condition_probe=arm,
        selection='fixed_step1000',scope='one_seed_condition_probe_pilot_validation_only')
    (ROOT/'configs'/f'{arm}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
locked=copy.deepcopy(prior['locked_file_sha256'])
for p in [Path(u['checkpoint']),Path(u['validation_metrics']),source_cfg,
          f/'scripts/run_frozen_adapter.py',f/'scripts/run_matched.py',
          f/'scripts/run_phase_objective.py',f/'scripts/reference/run_experiment_snapshot.py']:
    locked[str(p)]=sha(p)
for p in list((ROOT/'scripts').glob('*.py'))+list((ROOT/'configs').glob('*.yaml')):
    locked[str(p)]=sha(p)
lock=dict(scope='current_K2_path_and_alpha_diagnostic_then_conditional_one_seed_pilot',
    diagnostic_source=u,proxy=proxy,alphas=[0.0,0.25,0.5,0.75,1.0],
    locked_file_sha256=locked,training_seed=2026,evaluation_seed=2026,
    pilot_steps=1000,pilot_arms=['mixed_rgb','task_prediction'],
    training_gate='All historical/endpoint/frozen audits pass, finite values, and second-round changes reach gray, polarization parameters, prior condition and sampled prior on at least one locked image.',
    continuation_gate='After both 200-step stages: finite gradients, exactly 18 R2P optimizer states, frozen task/projection/feedback hashes, same initial tensors and logged schedule/RNG. Do not select by proxy score.',
    promotion_gate='Task-prediction arm must have positive four aggregate deltas versus both same-budget mixed-RGB arm and fixed G6 before considering more seeds; one seed is only an exploratory lead.',
    limits=['No test evaluations or test-based tuning.','No extra observations: task prediction is a learned representation of the same measurements.',
        'Predictor is frozen, condition is detached/clamped to [0,1], GT is never a condition.',
        '1000-step pilot is not evidence about convergence or all budgets.',
        'Original trained task has 3000 steps; adapter has additional 1000, no equal-total-budget ranking.',
        'Old task-feature feedback exists; this probes a different condition and staged topology, not proof of novelty.'])
assert not (ROOT/'protocol_lock.json').exists()
(ROOT/'protocol_lock.json').write_text(json.dumps(lock,indent=2))
print(json.dumps(dict(locked_files=len(locked),diagnostic_images=28,proxy_strata=proxy['stratum_counts'],pilot_steps=1000)))
