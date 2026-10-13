"""Reuse verified data/pretraining, not old joint-training weights or budgets."""
import csv,json,shutil
from pathlib import Path
from common import ROOT,sha,json_save

def prepare():
    old=ROOT.parent/'polar_float_bridge_joint_20261010'
    cfg=json.loads((old/'config/locked_run.json').read_text())
    for directory in ('config','diagnostics','logs','runtime/tmp','runtime/cache','pretraining/checkpoints',
                      'pretraining/metrics','weights/source_polarfree','downstream/config','downstream/checkpoints'):
        (ROOT/directory).mkdir(parents=True,exist_ok=True)
    for key in ('manifest','schedule','validation_proxy'):
        src=Path(cfg['paths'][key]);dst=ROOT/'config'/src.name
        shutil.copy2(src,dst);cfg['paths'][key]=str(dst)
    selected=ROOT/'pretraining/checkpoints/selected.pt'
    shutil.copy2(old/'pretraining/checkpoints/selected.pt',selected)
    assert sha(selected)==cfg['pretrain']['sha256']
    for relative in ('pretraining/metrics/final_test.json','pretraining/convergence.json',
                     'config/pidsr_data_manifest.json'):
        if (old/relative).exists():shutil.copy2(old/relative,ROOT/relative)
    for name,digest in cfg['task_initialization']['official_sha256'].items():
        src=Path(cfg['paths']['source_checkpoint']).parent/name;dst=ROOT/'weights/source_polarfree'/name
        assert sha(src)==digest;shutil.copy2(src,dst)
    # The official base configuration still strictly reads the same verified
    # official three files; both locations must have identical hashes.
    cfg['paths']['pretrained_r2p']=str(selected);cfg['paths']['output_root']=str(ROOT/'downstream')
    cfg['pretrain']['checkpoint']=str(selected)
    cfg['task_initialization']['archived_source_dir']=str(ROOT/'weights/source_polarfree')
    for key in ('budget','protocol_id','code_sha256','lambda_alignment'):cfg.pop(key,None)
    cfg['confidence']=dict(warmup_steps=500,calibration_training_crops=256)
    cfg['scope']='one arm: dedicated RGB residual correction with direct cross-round task/polar feature feedback'
    cfg['downstream']=dict(steps=12000,effective_batch=2,microbatch=1,minimum_steps=3000)
    cfg['optimization'].update(enhancer_freeze_steps=500)
    cfg['correction_design']=dict(state_channels=16,decoder_scales=['half','quarter'],zero_initial_output=True,
                                 output='T2=T1+0.25*tanh(learned_RGB_increment)',feature_injection_bound=.1,
                                 trusted_reference='C times previous projected task features',
                                 correction_context='raw RGB/observation polar summary/T1/I-T1/unscaled F1/polar features/C/1-C',
                                 quality_input_detached=True,quality_to_task_detached=True,
                                 first_and_second_backbone_shared=True,output_heads_shared=False,
                                 training_loss='official T1/3+2*T2/3; +0.1 bounded error-weighted T2 RGB MAE; quality NLL; alignment',
                                 error_weight='detach(clamp(1+T1 RGB error/(image mean error+1e-4),1,3)); normalized',
                                 no_extra_equivalent_residual_l1=True)
    cfg['mechanism_diagnostics']=dict(validation_frames=28,conditions=['normal','no_feedback','no_polar_correction','confidence_zero','confidence_one'],
                                     inference_interventions_only=True,test_used=False,checkpoint_selection=False)
    cfg['experiment_rationale']=dict(user_authorization='启动: implement and launch researched correction design',
                                    historical_test_exploration=True,matched_control=False,single_seed=True,
                                    previous_experiments_immutable=True,new_run_budget_hours_cap=24,
                                    planned_step_cap=12000,minimum_steps=3000)
    with Path(cfg['paths']['manifest']).open(newline='') as f:rows=list(csv.DictReader(f))
    counts={split:sum(r['split']==split for r in rows) for split in ('train','validation','test')}
    assert counts==dict(train=5440,validation=872,test=188)
    schedule=json.loads(Path(cfg['paths']['schedule']).read_text())
    assert len(schedule['items'])>=24000
    json_save(ROOT/'diagnostics/reuse_audit.json',dict(split_counts=counts,schedule_crops=len(schedule['items']),
              pretrain_sha256=sha(selected),official_sources=cfg['task_initialization']['official_sha256'],
              old_joint_weights_loaded=False,old_budget_reused=False))
    json_save(ROOT/'config/preflight.json',cfg)
    print(json.dumps(dict(prepared=True,counts=counts,crops=len(schedule['items']))),flush=True)

if __name__=='__main__':prepare()
