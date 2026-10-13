"""Freeze a time-budgeted study by extending existing data and training machinery."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'direct_t1_joint_20261008_memorysafe_v2'
BASE = SOURCE / 'scripts/reference/run_experiment_snapshot.py'
for name in ('run_matched.py', 'run_phase_objective.py'):
    shutil.copy2(SOURCE / 'scripts' / name, ROOT / 'scripts' / name)
text = BASE.read_text()
def replace(old, new):
    global text
    assert text.count(old) == 1, (old[:100], text.count(old))
    text = text.replace(old, new)

replace('feedback=None, project_features=True) -> JointOutput:\n        if feedback is not None:',
        'feedback=None, project_features=True, rgb_feature=None) -> JointOutput:\n'
        '        if rgb_feature is not None:\n'
        '            return self._confidence_forward(polar, rgb_feature, project_features)\n'
        '        if feedback is not None:')
replace('"params": list(model.parameters()),',
        '"params": [p for n, p in model.named_parameters() if not n.startswith("confidence_head.")],')
replace('"name": "tasknet_net_g",\n            },\n        ],',
        '"name": "tasknet_net_g",\n            },\n'
        '            {"params": list(model.confidence_head.parameters()),\n'
        '             "lr": 1e-4, "name": "quality_head"},\n        ],')
replace('        model.train()\n        task.eval()',
        '        model._confidence_step = step\n        model.train()\n        task.eval()')
replace('        loss = task_loss + coefficient * alignment_loss\n        loss.backward()',
        '        loss = task_loss + coefficient * alignment_loss + output.confidence_loss\n        loss.backward()')
replace('            "diffusion_seed": noise_seed,',
        '            "diffusion_seed": noise_seed,\n'
        '            "confidence_loss": float(output.confidence_loss.detach()),\n'
        '            "confidence_mean": float(output.confidence_map.mean()),\n'
        '            "predicted_error_mean": float(output.error_scale.detach().mean()),\n'
        '            "confidence_head_grad_norm": gradient_norm(model.confidence_head.parameters()),\n'
        '            "confidence_gamma": output.confidence_gamma,')
replace('            model.parameters(), float(config["optimization"]["grad_clip"]), error_if_nonfinite=True\n        )\n        torch.nn.utils.clip_grad_norm_(',
        '            [p for n, p in model.named_parameters() if not n.startswith("confidence_head.")],\n'
        '            float(config["optimization"]["grad_clip"]), error_if_nonfinite=True\n        )\n'
        '        torch.nn.utils.clip_grad_norm_(model.confidence_head.parameters(), 1., error_if_nonfinite=True)\n'
        '        torch.nn.utils.clip_grad_norm_(')
replace('        should_checkpoint = step % checkpoint_interval == 0 or step == steps',
        '        budget_stop = time.time() >= float(config["study"]["training_deadline_unix"])\n'
        '        should_checkpoint = step % checkpoint_interval == 0 or step == steps or budget_stop')
replace('    timing = {\n        "start_step": start_step,',
        '        if budget_stop:\n            break\n'
        '    timing = {\n        "actual_last_step": step,\n        "budget_stop": budget_stop,\n        "start_step": start_step,')
replace('        + "Initialization: PIDSR e42 + official TaskNet; resume uses last.pt.\\n"',
        '        + "Initialization: latest full direct-T1 K2; fresh Adam; quality head only is new.\\n"')
replace('        + f"Loss: 0.5 L1 + L2 + {coefficient:.9g} align1.\\n\\n"',
        '        + f"Loss: (L1+2L2)/3 + {coefficient:.9g} align1 + Lconf.\\n\\n"')
replace('        + "Best: round2 scene-macro PSNR, SSIM, earlier step; validation only.\\n"',
        '        + "Evaluation: fixed final common step; clock fallback only, never test selection.\\n"')
replace('        + "Against matched K2 control, only feedback usage differs.\\n",',
        '        + "Matched control shares source, schedule, optimizer and quality auxiliary training.\\n",')
(ROOT / 'scripts/reference').mkdir(exist_ok=True)
shutil.copy2(BASE, ROOT / 'scripts/reference/source_snapshot_unmodified.py')
(ROOT / 'scripts/reference/run_experiment_snapshot.py').write_text(text)

cfg = yaml.safe_load((SOURCE / 'config/study.yaml').read_text())
sys.path.insert(0, cfg['paths']['source_root'])
from rgb_final_confirmation.data import load_manifest, select_rows, build_training_schedule
rows = select_rows(load_manifest(cfg['paths']['manifest']), 'train')
schedule = build_training_schedule(rows, seed=2026, total_steps=16000, batch_size=2, crop_size=128)
assert len(rows) == 5440
(ROOT / 'config').mkdir(exist_ok=True)
schedule_path = ROOT / 'config/training_schedule_16000.json'
schedule_path.write_text(json.dumps(schedule))
from datetime import datetime
deadline = datetime.fromisoformat('2026-10-09T00:55:33+00:00').timestamp()
cfg['paths']['schedule'] = str(schedule_path)
cfg['optimization'].update(total_steps=16000, validation_steps=[200, 16000], checkpoint_interval=1000)
cfg['study'].update(experiment='T1_quality_confidence_overnight', fixed_budget=16000,
                    source_checkpoint=str(SOURCE / 'run/checkpoints/step_10000.pt'),
                    source_sha256='eb17a5537bcd26583b73670a34a258f948056935f851b42c7a44eb62e1dd15ea',
                    source_training_steps=13000, optimizer_initialization='fresh_Adam',
                    deadline_unix=deadline, evaluation_reserve_seconds=7800,
                    training_deadline_unix=deadline - 7800,
                    selection='fixed_step16000; time-only fallback to latest common saved step')
cfg['confidence'] = dict(mode='dual', tau=.1, initial_error=.05, warmup_steps=500,
                         head_parameters=23353, supervised_only=True, detached_gate=True,
                         calibration_train_crops=256)
for name, mode in [('A2_dual', 'dual'), ('A0_direct', 'direct')]:
    arm = copy.deepcopy(cfg)
    arm['confidence']['mode'] = mode
    arm['paths']['output_root'] = str(ROOT / 'arms' / name / 'run')
    arm['study']['unit'] = name
    (ROOT / 'config' / f'{name}.yaml').write_text(yaml.safe_dump(arm, sort_keys=False))
counts = {}
for item in schedule['items']:
    i = item['sample_index']; counts[i] = counts.get(i, 0) + 1
lock = dict(train_frames=5440, train_patches=32000, unique_train_frames=len(counts),
            min_visits=min(counts.values()), max_visits=max(counts.values()),
            deadline_unix=deadline, target_steps=16000, order=['A2_dual', 'A0_direct'],
            evaluation_reserve_seconds=7800, source_sha256=cfg['study']['source_sha256'],
            mandatory_validation_frames=872, mandatory_test_frames=188,
            selection='common saved checkpoint by time and step only; no metric selection',
            manifest_sha256=hashlib.sha256(Path(cfg['paths']['manifest']).read_bytes()).hexdigest(),
            schedule_sha256=hashlib.sha256(schedule_path.read_bytes()).hexdigest())
assert len(counts) == 5440
(ROOT / 'protocol_lock.json').write_text(json.dumps(lock, indent=2))
print(json.dumps(lock), flush=True)
