"""Adapt the project's CPU archive audit to the two-stage/action-cost layout."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math,statistics

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
receipt=read(ROOT/'archive_receipt.json');assert receipt['all_hashes_match']
manifest={r['path']:r for r in read(ART/'archive_manifest.json')['files']}
cfg=read(ART/'config/locked_run.json');selected=read(ART/'downstream/selection.json')
assert selected['candidate_steps']==0 and selected['judgement_steps']==3000 and not selected['test_used']
assert selected['sha256']==manifest['downstream/checkpoints/final.pt']['sha256']=='279d0a329d0efebee9678c2156bf3e7b60c0646e75a5f37f8015c7b100f9e9a2'
assert cfg['reuse']['sha256']==manifest['weights/reused_final.pt']['sha256']=='cecdcfa47d290b0a2d6b662b96bb4407a7043ef5789e9e87d70768f2c683916d'
assert manifest['pretraining/checkpoints/selected.pt']['sha256']=='a4ecebacba5cfc25546320974a9c37f1cc220c685f1318b3cc290ae4a68e9554'
for name,digest in cfg['task_initialization']['official_sha256'].items():assert manifest['weights/source_polarfree/'+name]['sha256']==digest
for name,digest in cfg['code_sha256'].items():assert manifest[name]['sha256']==digest,name
assert hashlib.sha256((ART/'config/locked_run.json').read_bytes()).hexdigest()==read(ART/'config/lock_receipt.json')['locked_sha256']
for name,entry in cfg.get('runtime_source_evidence',{}).items():assert manifest[entry['archived_copy']]['sha256']==entry['sha256']

weight_audit=read(ART/'recovery/final_cpu_weight_audit.json')
assert weight_audit['protocol_id']==cfg['protocol_id'] and not weight_audit['gpu_used']
assert weight_audit['source_checkpoint_sha256']==manifest['weights/reused_final.pt']['sha256']
assert weight_audit['final_checkpoint_sha256']==manifest['downstream/checkpoints/final.pt']['sha256']
assert weight_audit['frozen_model_tensors_equal']==518 and weight_audit['net_g_tensors_equal']==436
assert not weight_audit['frozen_code_mismatches']
for name,digest in cfg['configuration_provenance_sha256'].items():assert manifest[name]['sha256']==digest,name
weight_audit['matches_archived_weight_bytes']=True
output=ROOT/'interpretation';output.mkdir(exist_ok=True)
(output/'final_cpu_weight_audit.json').write_text(json.dumps(weight_audit,indent=2))
summary=dict(protocol_id=cfg['protocol_id'],checkpoint_sha256=selected['sha256'],source_weights_match=True,
 frozen_code_match=True,frozen_model_tensor_keys=518,all_frozen_model_tensors_equal_reused_source=True,
 all_net_g_tensors_equal_reused_source=True,candidate_and_q_frozen_entire_run=True,gpu_used_for_this_analysis=False,
 cpu_tensor_audit_location='server CPU with CUDA disabled; byte hashes match locally verified archive',
 source_model_keys=526,final_model_keys=526,new_guidance_keys=0,splits={})

with (ART/'config/manifest.csv').open(encoding='utf-8',newline='') as f:data=list(csv.DictReader(f))
roles=read(ART/'config/training_role_manifest.json');train_ids={r['sample_id'] for r in data if r['split']=='train'}
assert len(train_ids)==5440 and set().union(*(set(x) for x in roles['sample_ids'].values()))==train_ids
assert all(not(set(roles['scene_roles'][a])&set(roles['scene_roles'][b])) for a in roles['scene_roles'] for b in roles['scene_roles'] if a!=b)
training={}
for phase,rate in [('judgement',1e-4)]:
    total=cfg['staged']['steps'][phase]
    with (ART/f'downstream/logs/{phase}_train.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    assert [int(r['step']) for r in rows]==list(range(1,total+1))
    seen=Counter(i for r in rows for i in r['sample_ids'].split(';'))
    assert set(seen)==set(roles['sample_ids'][phase]) and sum(seen.values())==2*total
    for r in rows:
        assert float(r['learning_rate'])==rate and float(r['core_lr'])==float(r['task_lr'])==0
        assert all(math.isfinite(float(r[k])) for k in ('loss','grad_norm','seconds'))
    training[phase]=dict(steps=total,unique_images=len(seen),sampled_crops=2*total,learning_rate=rate)
cal=read(ART/'downstream/config/action_calibration.json')
for role,key in [('bias','bias_sample_ids'),('safety','safety_sample_ids')]:assert set(cal[key])==set(roles['sample_ids'][role])
assert not cal['validation_used'] and not cal['test_used'];assert cal['chosen']['coverage']==0
summary['training']=dict(phases=training,role_image_counts=roles['sample_counts'],all5440_manifest_and_roles_verified=True,current_fit_and_calibration_images=1800,candidate_role_images_not_updated=3640,
                        inherited_weights_previously_saw_roles=True,rates_and_freeze_verified=True)
summary['calibration']=cal

output=ROOT/'interpretation';output.mkdir(exist_ok=True)
for split,count in [('validation',872),('test',188)]:
    x=read(ART/f'downstream/metrics/final_{split}.json');rows=x['per_image'];gate=x['action_acceptance'];grows=gate['per_image']
    assert x['frames']==len(rows)==len({r['sample_id'] for r in rows})==count
    assert {r['sample_id'] for r in rows}=={r['sample_id'] for r in data if r['split']==split}
    assert x['original_size'] and not x['tiling'] and not x['test_used_for_selection']
    assert x['checkpoint_sha256']==selected['sha256'] and x['protocol_id']==cfg['protocol_id']
    assert all(r['round1']==r['round2'] and r['iteration_gain']==dict(psnr=0.,ssim=0.) for r in rows)
    for k in ('round1','round2','ungated_candidate','iteration_gain'):
        for metric in ('psnr','ssim'):
            values=[r[k][metric] for r in rows];assert all(math.isfinite(v) for v in values)
            assert abs(statistics.mean(values)-x[k]['image_mean_'+metric])<1e-10
    for k,total in gate['totals'].items():assert sum(r[k] for r in grows)==total
    assert gate['totals']['accepted']==0 and gate['protected_pixel_fraction']==1.
    old=read(ROOT.parent/'polar_error_guided_short_20261012/artifacts/downstream/metrics'/f'final_{split}.json')
    assert [r['sample_id'] for r in rows]==[r['sample_id'] for r in old['per_image']]
    assert all(r['round1']==o['round1'] and r['ungated_candidate']==o['ungated_candidate'] for r,o in zip(rows,old['per_image']))
    dp=[r['ungated_candidate']['psnr']-r['round1']['psnr'] for r in rows]
    ds=[r['ungated_candidate']['ssim']-r['round1']['ssim'] for r in rows]
    summary['splits'][split]=dict(frames=count,round1=x['round1'],round2=x['round2'],ungated_candidate=x['ungated_candidate'],iteration_gain=x['iteration_gain'],
        candidate_psnr_gain=x['ungated_candidate']['image_mean_psnr']-x['round1']['image_mean_psnr'],candidate_ssim_gain=x['ungated_candidate']['image_mean_ssim']-x['round1']['image_mean_ssim'],
        candidate_images_improved=sum(v>0 for v in dp),candidate_images_worsened=sum(v<0 for v in dp),candidate_both_improved=sum(a>0 and b>0 for a,b in zip(dp,ds)),
        candidate_scenes_improved=sum(r['ungated_candidate']['psnr']>r['round1']['psnr'] for r in x['per_scene']),per_scene=x['per_scene'],
        action_acceptance={k:v for k,v in gate.items() if k!='per_image'},confidence={k:v for k,v in x['confidence'].items() if k!='per_image'},
        t1_matches_previous_full_evaluation=True,all_images_t2_equals_t1=True)
    with (output/f'{split}_per_image.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['sample_id','scene','t1_psnr','t2_psnr','candidate_psnr','t1_ssim','t2_ssim','candidate_ssim','psnr_gain','ssim_gain','mask_mean']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r,g in zip(rows,grows):writer.writerow(dict(sample_id=r['sample_id'],scene=r['scene'],t1_psnr=r['round1']['psnr'],t2_psnr=r['round2']['psnr'],candidate_psnr=r['ungated_candidate']['psnr'],t1_ssim=r['round1']['ssim'],t2_ssim=r['round2']['ssim'],candidate_ssim=r['ungated_candidate']['ssim'],psnr_gain=r['iteration_gain']['psnr'],ssim_gain=r['iteration_gain']['ssim'],mask_mean=g['mask_mean']))
    assert len(list((ART/'downstream/examples'/split).glob('*.png')))==40
probes=[]
reference=read(ART/'downstream/metrics/probe_initial.json')['per_image']
reference=[(r['sample_id'],r['t1_tensor_sha256'],r['candidate_tensor_sha256']) for r in reference]
for p in (ART/'downstream/metrics').glob('probe_*.json'):
    x=read(p);assert x['training_role']=='bias' and x['frames']==32 and x['t1_and_candidate_hashes_fixed']
    assert x['all_non_judgement_tensors_equal_source']==518
    assert [(r['sample_id'],r['t1_tensor_sha256'],r['candidate_tensor_sha256']) for r in x['per_image']]==reference
    probes.append(p.name)
assert len(probes)==8;summary['first_round_all_probes_exactly_unchanged']=True
summary['candidate_all_probes_exactly_unchanged']=True
mechanism=read(ART/'downstream/metrics/mechanism_validation.json');assert mechanism['frames']==len(mechanism['per_image'])==28 and not mechanism['test_used']
summary['mechanism']={k:v for k,v in mechanism.items() if k!='per_image'}
summary['evaluation_completed_unix']=read(ART/'status.json')['time_unix'];summary['budget']=read(ART/'budget.json')
assert summary['budget']['start_unix']==1791810276.3391461 and summary['budget']['deadline_unix']==1791839076.3391461
assert summary['evaluation_completed_unix']<summary['budget']['deadline_unix']
summary['saved_action_example_scenes']=16
summary['saved_confidence_example_scenes']=len(list((ART/'downstream/metrics/confidence_examples').glob('*/*.npz')))
assert summary['saved_confidence_example_scenes']==16
summary['implementation_and_experiment_completed']=True;summary['method_iteration_improvement_observed']=False
source_oracle=read(ART/'diagnostics/source_oracle_validation872.json')
final_oracle=read(ART/'diagnostics/final_oracle_validation872.json')
assert source_oracle['frames']==final_oracle['frames']==872 and final_oracle['t1_all_float_hashes_equal_source']
assert source_oracle['oracle_uses_GT'] and final_oracle['oracle_uses_GT'] and not source_oracle['test_used'] and not final_oracle['test_used']
assert len(source_oracle['per_image'])==len(final_oracle['per_image'])==872
summary['oracle_diagnostics']=dict(source_means=source_oracle['means'],final_means=final_oracle['means'],oracle_deployable=False,frames=872,scales=[4,16,32],same_t1_float_hashes=True)
summary['repair_need']=dict(bce_image_mean=statistics.mean(r['repair_need_bce'] for r in final_oracle['per_image']),accuracy_image_mean=statistics.mean(r['repair_need_accuracy'] for r in final_oracle['per_image']),target='train-only local-error proxy; not reflection GT',probability_calibration_established=False)
summary['archived_files_verified']=receipt['files_verified']
summary['predictability_diagnostic']=read(ART/'diagnostics/source_predictability_validation872.json')['means']
summary['training_trigger']=read(ART/'diagnostics/training_trigger.json')
summary['failure_archive_verification']={}
for failure in ('recording_failure_20261012','reference_failure_20261012'):
    failure_receipt=read(ROOT/'failures'/failure/'archive_receipt.json')
    assert failure_receipt['all_hashes_match'] and not failure_receipt['archive_completed']
    summary['failure_archive_verification'][failure]=failure_receipt
for split in ('validation','test'):
    metric=read(ART/f'downstream/metrics/final_{split}.json')
    assert len(metric['action_acceptance']['all_nonzero_action_predictions'])==metric['frames']
    assert metric['action_acceptance']['candidate_hashes_match_source_validation']==(split=='validation')
    action_summary=[]
    for j in range(3):
        rows=[r['actions'][j] for r in metric['action_acceptance']['all_nonzero_action_predictions']]
        values=[r['gain_spearman'] for r in rows if r['gain_spearman'] is not None]
        action_summary.append(dict(action=rows[0]['action'],predicted_mean_gain=statistics.mean(r['predicted_mean_gain'] for r in rows),actual_mean_gain=statistics.mean(r['actual_mean_gain'] for r in rows),cost_mae=statistics.mean(r['cost_mae'] for r in rows),gain_spearman_image_mean=statistics.mean(values) if values else None))
    summary['splits'][split]['independent_nonzero_action_scores']=action_summary

(output/'result_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(source_and_final_weights_verified=True,frozen_keys=518,phases=training,all_images_t2_equals_t1=True,roles=roles['sample_counts'],original_size_frames=1060,gpu_used=False)))
