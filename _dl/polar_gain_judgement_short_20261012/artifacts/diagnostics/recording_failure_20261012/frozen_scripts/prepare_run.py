"""Reuse verified data/pretraining, not old joint-training weights or budgets."""
import csv,json,shutil,yaml
from pathlib import Path
from common import ROOT,sha,json_save

def prepare():
    old=ROOT.parent/'polar_residual_feedback_joint_20261010'
    cfg=json.loads((old/'config/locked_run.json').read_text())
    old_protocol=cfg['protocol_id']
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
    source=ROOT/'source'
    shutil.copytree(cfg['paths']['source_root'],source,ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
    cfg['paths']['source_root']=str(source)
    base_file=ROOT/'config/base_server.yaml'
    raw_base=yaml.safe_load(Path(cfg['paths']['base_config']).read_text())
    raw_base['project_root']=str(ROOT)
    base_file.write_text(yaml.safe_dump(raw_base,sort_keys=False),encoding='utf-8')
    cfg['paths']['base_config']=str(base_file)
    cfg['pretrain']['checkpoint']=str(selected)
    cfg['task_initialization']['archived_source_dir']=str(ROOT/'weights/source_polarfree')
    for key in ('budget','protocol_id','code_sha256','lambda_alignment'):cfg.pop(key,None)
    # The fixed T1/H pair inherits its already training-calibrated temperature.
    cfg['confidence']['reused_from_previous_final']=True
    warmstart=ROOT/'weights/reused_final.pt'
    shutil.copy2(old/'downstream/checkpoints/final.pt',warmstart)
    assert sha(warmstart)=='ec7f0e6e79d0dcf21c536dc71958d44514bd7e78a8624bb936f16c02c37e81e8'
    cfg['reuse']=dict(enabled=True,checkpoint=str(warmstart),sha256=sha(warmstart),
         source_run=str(old),source_step=12000,source_protocol_id=old_protocol,
         fixed_first_round=True,reuse_optimizer=False,new_initialization_only=['local_acceptance'],
         first_500_steps='only new judgement module',later='second-round feedback interfaces at 1e-5; judgement 1e-4',
         frozen_entire_run=['r2p core and old adapters','PolarFree net_g','prior networks','H','first-round projections'])
    cfg['task_initialization']['kind']='verified_previous_final'
    cfg['scope']='one arm: local need/benefit judgement and final RGB update gate'
    cfg['downstream']=dict(steps=6000,effective_batch=2,microbatch=1,minimum_steps=3000)
    cfg['optimization'].update(correction_lr=0.,task_lr=0.,interfaces_lr=1e-5,judgement_lr=1e-4,
                               enhancer_freeze_steps='entire run',task_freeze_steps='entire run')
    cfg['correction_design']=dict(state_channels=16,decoder_scales=['half','quarter'],zero_initial_output=True,
                                 output='T2=T1+0.25*tanh(learned_RGB_increment)',feature_injection_bound=.1,
                                 trusted_reference='C times previous projected task features',
                                 correction_context='raw RGB/observation polar summary/T1/I-T1/unscaled F1/polar features/C/1-C',
                                 quality_input_detached=True,quality_to_task_detached=True,
                                 first_and_second_backbone_shared=True,output_heads_shared=False,
                                 training_loss='official T1/3+2*T2/3; +0.1 bounded error-weighted T2 RGB MAE; quality NLL; alignment',
                                 error_weight='detach(clamp(1+T1 RGB error/(image mean error+1e-4),1,3)); normalized',
                                 no_extra_equivalent_residual_l1=True)
    cfg['local_acceptance']=dict(grid=4,judgement_input_detached=True,judgement_to_task_detached=True,
           outputs=['need','better','neutral','worse'],training_warmup=500,transition_steps=1000,
           good_threshold_rule='256 initial training crops, local error 25th percentile',
           benefit_margin_rule='max(.05*initial median local error,1e-4)',
           local_error='RGB MAE+.25 edge MAE pooled4',candidate_loss_weight=.25,
           preservation_loss_weight=.1,judgement_loss_weight=.05,
           decision='need>.5 and better>neutral and better>worse',
           magnitude='need*(better-worse), bilinear; allow nearest enforces exact zeros',
           output='T2=where(M=0,T1,T1+M*RGB_increment)',test_used=False)
    cfg['correction_design']['training_loss']='T1/3+2*gated_T2/3; .1 weighted gated MAE; .25 ungated candidate local error; .1 good-region candidate change; H NLL; .05 need/benefit classification; alignment'
    cfg['mechanism_diagnostics']=dict(validation_frames=28,conditions=['normal','no_feedback','no_polar_correction','confidence_zero','confidence_one','mask_zero','mask_one'],
                                     inference_interventions_only=True,test_used=False,checkpoint_selection=False)
    cfg['experiment_rationale']=dict(user_authorization='去做吧: implement and run proposed local acceptance mechanism',
                                    historical_test_exploration=True,matched_control=False,single_seed=True,
                                    previous_experiments_immutable=True,new_run_budget_hours_cap=24,
                                    planned_step_cap=6000,minimum_steps=3000,
                                    continuation_from_previous_final=True)
    with Path(cfg['paths']['manifest']).open(newline='') as f:rows=list(csv.DictReader(f))
    counts={split:sum(r['split']==split for r in rows) for split in ('train','validation','test')}
    assert counts==dict(train=5440,validation=872,test=188)
    schedule=json.loads(Path(cfg['paths']['schedule']).read_text())
    assert len(schedule['items'])>=24000
    json_save(ROOT/'diagnostics/reuse_audit.json',dict(split_counts=counts,schedule_crops=len(schedule['items']),
              pretrain_sha256=sha(selected),official_sources=cfg['task_initialization']['official_sha256'],
              previous_verified_final_loaded=True,untracked_historical_weights_loaded=False,old_budget_reused=False))
    json_save(ROOT/'config/preflight.json',cfg)
    print(json.dumps(dict(prepared=True,counts=counts,crops=len(schedule['items']))),flush=True)

if __name__=='__main__':prepare()
