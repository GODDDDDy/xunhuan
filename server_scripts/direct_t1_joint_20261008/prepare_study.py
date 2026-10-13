"""Create the new study with existing data/schedule machinery and trained K1."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
SOURCE = PROJECT/'normalized_threeway_20261002'
assert (ROOT/'scripts/test_completion.py').is_file(), 'Copy test_completion.py with the three study scripts before preparation'
for name in ['run_matched.py', 'run_phase_objective.py', 'reference/run_experiment_snapshot.py']:
    dst = ROOT/'scripts'/name
    dst.parent.mkdir(parents=True, exist_ok=True)
    assert not dst.exists(), str(dst)
    shutil.copy2(SOURCE/'scripts'/name, dst)
snapshot=ROOT/'scripts/reference/run_experiment_snapshot.py'
original=snapshot.read_text()
shutil.copy2(snapshot,snapshot.with_name('run_experiment_snapshot_original.py'))
needle='''            metrics = validate(
                model, task, validation_loader, device, int(config["seed"])
            )'''
replacement='''            metrics = validate_with_optimizer_offload(
                model, task, optimizer, validation_loader, device, int(config["seed"])
            )'''
assert original.count(needle)==1
revised=original.replace(needle,replacement)
# load_state_dict copies model weights and moves Adam state to each parameter's
# device. Keep the serialized checkpoint copy on CPU, including during evaluation.
for variable in ['latest_path','checkpoint_path']:
    old=f'state = torch.load({variable}, map_location=device, weights_only=False)'
    new=f'state = torch.load({variable}, map_location="cpu", weights_only=False)'
    assert revised.count(old)==1
    revised=revised.replace(old,new)
snapshot.write_text(revised)
cfg = yaml.safe_load((SOURCE/'configs/K2_seed2026.yaml').read_text())
sys.path.insert(0, cfg['paths']['source_root'])
from rgb_final_confirmation.data import load_manifest, select_rows, build_training_schedule
rows = select_rows(load_manifest(cfg['paths']['manifest']), 'train')
assert len(rows) == 5440
schedule = build_training_schedule(rows, seed=2026, total_steps=10000, batch_size=2, crop_size=128)
cfg['paths']['schedule'] = str(ROOT/'config/training_schedule_10000.json')
cfg['paths']['output_root'] = str(ROOT/'run')
cfg['optimization'].update(total_steps=10000, validation_steps=[0,200,1000,3000,6000,10000], checkpoint_interval=1000)
cfg['recurrent']['condition_mode'] = 'direct_T1_image'
checkpoint = SOURCE/'runs/K1/seed_2026/checkpoints/step_3000.pt'
source_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert source_sha == 'f87e8b75d1335c7ae979ee563cac14270258c130ac2953a5903b7eb7de647b64'
cfg['study'].update(experiment='K1_warmstart_direct_T1_joint', fixed_budget=10000,
    selection='fixed_additional_step10000', scope='single_seed_full_training_pool_direct_T1_joint',
    source_K1_checkpoint=str(checkpoint), source_K1_sha256=source_sha,
    source_K1_validation=str(SOURCE/'runs/K1/seed_2026/metrics/step_3000_validation.json'),
    optimizer_initialization='fresh_Adam', confidence_enabled=False)
(ROOT/'config').mkdir(exist_ok=True)
(ROOT/'config/training_schedule_10000.json').write_text(json.dumps(schedule,indent=2))
(ROOT/'config/study.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
counts = {}
for item in schedule['items']:
    i=item['sample_index']; counts[i]=counts.get(i,0)+1
lock = {'experiment':cfg['study']['experiment'], 'seed':2026, 'source_K1_checkpoint':str(checkpoint),
        'source_K1_sha256':source_sha, 'source_training_steps':3000, 'additional_training_steps':10000,
        'total_training_steps':13000, 'batch_size':2, 'crop_size':128, 'train_frames':len(rows),
        'patches':len(schedule['items']), 'unique_train_frames':len(counts),
        'min_frame_visits':min(counts.values()), 'max_frame_visits':max(counts.values()),
        'optimizer_initialization':'fresh_Adam', 'alignment':'first_round_three_level_cosine_stop_target',
        'memory_revision':'CPU_checkpoint_read_validation_Adam_offload_and_poststep_grad_release',
        'failed_previous_run':str(PROJECT/'direct_t1_joint_20261008'),
        'interrupted_previous_run':str(PROJECT/'direct_t1_joint_20261008_memorysafe'),
        'lambda_alignment':cfg['study']['fixed_lambda'], 'round_loss_weights':[1/3,2/3],
        'second_R2P_condition':'clamp(T1,0,1), without detach', 'TaskNet_RGB':'original mixed RGB both rounds',
        'internal_feature_feedback':False, 'confidence':False, 'prior_parameters_frozen':True,
        'final_evaluation':'872 full-resolution validation followed by all 188 test frames',
        'test_required_for_completion':True, 'checkpoint_selection':'fixed_additional_step10000',
        'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'scripts').rglob('*.py'))},
        'config_sha256':hashlib.sha256((ROOT/'config/study.yaml').read_bytes()).hexdigest(),
        'manifest_sha256':hashlib.sha256(Path(cfg['paths']['manifest']).read_bytes()).hexdigest(),
        'schedule_sha256':hashlib.sha256(Path(cfg['paths']['schedule']).read_bytes()).hexdigest()}
assert lock['unique_train_frames']==5440 and lock['patches']==20000
(ROOT/'protocol_lock.json').write_text(json.dumps(lock,indent=2))
print(json.dumps({k:v for k,v in lock.items() if k not in ['code_sha256']},indent=2))
