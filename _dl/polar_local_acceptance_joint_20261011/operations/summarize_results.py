"""CPU-only audit/summary of the completed region protection experiment."""
from pathlib import Path
from collections import Counter
import csv, hashlib, json, math, statistics
import torch

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts'
def read(path): return json.loads(path.read_text(encoding='utf-8'))
receipt=read(ROOT/'archive_receipt.json')
assert receipt['all_hashes_match']
manifest={x['path']:x for x in read(ART/'archive_manifest.json')['files']}
cfg=read(ART/'config/locked_run.json')
selected=read(ART/'downstream/selection.json')
assert cfg['reuse']['enabled'] and cfg['reuse']['fixed_first_round']
assert selected['step']==cfg['downstream']['steps']==6000 and not selected['test_used']
assert selected['sha256']==manifest['downstream/checkpoints/final.pt']['sha256']=='e8a635e052ec86f9abddeaa6731a3cb85533420aa78d3b0ff52fbac5bd38e299'
assert manifest['weights/reused_final.pt']['sha256']==cfg['reuse']['sha256']=='ec7f0e6e79d0dcf21c536dc71958d44514bd7e78a8624bb936f16c02c37e81e8'
for name,digest in cfg['task_initialization']['official_sha256'].items():
    assert manifest['weights/source_polarfree/'+name]['sha256']==digest
assert manifest['pretraining/checkpoints/selected.pt']['sha256']==cfg['pretrain']['sha256']
for name,digest in cfg['code_sha256'].items():
    assert name in manifest and manifest[name]['sha256']==digest,name
assert hashlib.sha256((ART/'config/locked_run.json').read_bytes()).hexdigest()==read(ART/'config/calibrated_lock_receipt.json')['final_configuration_sha256']

source=torch.load(ART/'weights/reused_final.pt',map_location='cpu',weights_only=False)
final=torch.load(ART/'downstream/checkpoints/final.pt',map_location='cpu',weights_only=False)
frozen_keys=[n for n in source['r2p_and_projections'] if not n.startswith('feedback.')]
assert all(torch.equal(v,source['tasknet_net_g'][n]) for n,v in final['tasknet_net_g'].items())
assert all(torch.equal(final['r2p_and_projections'][n],source['r2p_and_projections'][n]) for n in frozen_keys)
assert not torch.cuda.is_initialized()
summary=dict(protocol_id=cfg['protocol_id'],checkpoint_step=6000,checkpoint_sha256=selected['sha256'],
             archive_files_verified=receipt['files_verified'],source_weights_match=True,frozen_code_match=True,
             frozen_model_tensor_keys=len(frozen_keys),all_net_g_tensors_equal_reused_source=True,
             all_frozen_model_tensors_equal_reused_source=True,gpu_used_for_this_analysis=False,splits={})
del source,final

with (ART/'config/manifest.csv').open(encoding='utf-8',newline='') as f:
    manifest_rows=list(csv.DictReader(f))
with (ART/'downstream/logs/train.csv').open(encoding='utf-8',newline='') as f:
    train=list(csv.DictReader(f))
assert [int(r['step']) for r in train]==list(range(1,6001))
seen=Counter(s for r in train for s in r['sample_ids'].split(';'))
assert set(seen)=={r['sample_id'] for r in manifest_rows if r['split']=='train'}
assert len(seen)==5440 and sum(seen.values())==12000
for r in train:
    assert all(math.isfinite(float(r[k])) for k in ('loss','task','quality','grad_norm','judge_loss','candidate_loss','preservation_loss'))
    assert r['tasknet_trainable']=='False' and r['reused_first_round_frozen']=='True'
    assert float(r['task_lr'])==float(r['correction_lr'])==0
    assert float(r['interfaces_lr'])==(0 if int(r['step'])<=500 else 1e-5)
    assert float(r['judgement_lr'])==1e-4
summary['training']=dict(steps=6000,unique_images=len(seen),sampled_crops=sum(seen.values()),
                         min_visits=min(seen.values()),max_visits=max(seen.values()),rates_and_freeze_verified=True)

for split,count in [('validation',872),('test',188)]:
    x=read(ART/f'downstream/metrics/final_{split}.json')
    rows=x['per_image'];gate=x['local_acceptance'];grows=gate['per_image']
    assert x['frames']==len(rows)==len({r['sample_id'] for r in rows})==count
    assert {r['sample_id'] for r in rows}=={r['sample_id'] for r in manifest_rows if r['split']==split}
    assert x['original_size'] and not x['tiling'] and not x['test_used_for_selection']
    assert x['checkpoint_sha256']==selected['sha256'] and x['protocol_id']==cfg['protocol_id']
    assert [r['sample_id'] for r in grows]==[r['sample_id'] for r in rows]
    for k in ('round1','round2','ungated_candidate','iteration_gain'):
        for m in ('psnr','ssim'):
            values=[r[k][m] for r in rows]
            assert all(math.isfinite(v) for v in values)
            assert abs(statistics.mean(values)-x[k]['image_mean_'+m])<1e-10
    for m in ('psnr','ssim'):
        assert all(abs(r['round2'][m]-r['round1'][m]-r['iteration_gain'][m])<1e-10 for r in rows)
    for k,total in gate['totals'].items(): assert sum(r[k] for r in grows)==total,k
    assert gate['all_protected_pixels_exactly_preserved'] and gate['target_used_only_after_prediction']
    assert gate['calibration_not_from_test']
    matrix=gate['benefit_confusion'];cells=gate['totals']['cells']
    assert sum(map(sum,matrix))==cells
    psnr=[r['iteration_gain']['psnr'] for r in rows];ssim=[r['iteration_gain']['ssim'] for r in rows]
    previous=read(ROOT.parent/'polar_residual_feedback_joint_20261010/artifacts/downstream/metrics'/f'final_{split}.json')
    assert [r['sample_id'] for r in rows]==[r['sample_id'] for r in previous['per_image']]
    assert all(r['round1']==p['round1'] for r,p in zip(rows,previous['per_image']))
    confidence={k:v for k,v in x['confidence'].items() if k!='per_image'}
    summary['splits'][split]=dict(frames=count,round1=x['round1'],round2=x['round2'],
        ungated_candidate=x['ungated_candidate'],iteration_gain=x['iteration_gain'],
        psnr_improved=sum(v>0 for v in psnr),psnr_worsened=sum(v<0 for v in psnr),
        psnr_unchanged=sum(v==0 for v in psnr),ssim_improved=sum(v>0 for v in ssim),
        ssim_worsened=sum(v<0 for v in ssim),both_improved=sum(a>0 and b>0 for a,b in zip(psnr,ssim)),
        median_psnr_gain=statistics.median(psnr),maximum_psnr_gain=max(psnr),minimum_psnr_gain=min(psnr),
        scenes=len(x['per_scene']),psnr_scenes_improved=sum(r['iteration_gain']['psnr']>0 for r in x['per_scene']),
        per_scene=x['per_scene'],confidence=confidence,local_acceptance={k:v for k,v in gate.items() if k!='per_image'},
        need_accuracy_cell_weighted=sum(r['need_accuracy']*r['cells'] for r in grows)/cells,
        need_majority_baseline=max(gate['totals']['good'],gate['totals']['difficult'])/cells,
        benefit_accuracy=sum(matrix[i][i] for i in range(3))/cells,
        benefit_majority_baseline=max(map(sum,matrix))/cells,
        average_mask=statistics.mean(r['mask_mean'] for r in grows),
        t1_local_error=statistics.mean(r['t1_error'] for r in grows),
        candidate_local_error=statistics.mean(r['proposal_error'] for r in grows),
        final_local_error=statistics.mean(r['final_error'] for r in grows),t1_matches_previous_full_evaluation=True)
    out=ROOT/'interpretation'/f'{split}_per_image.csv'
    with out.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sample_id','scene','t1_psnr','t2_psnr','candidate_psnr','t1_ssim','t2_ssim','candidate_ssim','psnr_gain','ssim_gain','mask_mean'])
        w.writeheader()
        for r,g in zip(rows,grows):
            w.writerow(dict(sample_id=r['sample_id'],scene=r['scene'],t1_psnr=r['round1']['psnr'],
                t2_psnr=r['round2']['psnr'],candidate_psnr=r['ungated_candidate']['psnr'],
                t1_ssim=r['round1']['ssim'],t2_ssim=r['round2']['ssim'],candidate_ssim=r['ungated_candidate']['ssim'],
                psnr_gain=r['iteration_gain']['psnr'],ssim_gain=r['iteration_gain']['ssim'],mask_mean=g['mask_mean']))

mechanism=read(ART/'downstream/metrics/mechanism_validation.json')
assert mechanism['frames']==len(mechanism['per_image'])==28 and not mechanism['test_used']
summary['mechanism']={k:v for k,v in mechanism.items() if k!='per_image'}
for x in summary['mechanism']['means'].values():
    x['psnr_change_from_normal']=x['psnr']-summary['mechanism']['means']['normal']['psnr']
initial=read(ART/'downstream/metrics/system_step_0.json')
for step in (500,1000,2000,3000,4000,5000,6000):
    probe=read(ART/f'downstream/metrics/system_step_{step}.json')
    assert probe['first_round_outputs_exactly_unchanged']
    assert [r['t1_tensor_sha256'] for r in probe['per_image']]==[r['t1_tensor_sha256'] for r in initial['per_image']]
summary['first_round_all_probes_exactly_unchanged']=True
summary['evaluation_completed_unix']=read(ART/'status.json')['time_unix']
assert summary['evaluation_completed_unix']<cfg['budget']['deadline_unix']
summary['budget']=cfg['budget']
for kind in ('acceptance_examples','confidence_examples'):
    assert len(list((ART/'downstream/metrics'/kind).glob('*/*.npz')))==16
(ROOT/'interpretation/result_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(files_verified=receipt['files_verified'],frozen_parameters_equal=True,
    training=summary['training'],splits={s:{k:v[k] for k in ('psnr_improved','psnr_worsened','ssim_improved','both_improved','psnr_scenes_improved','average_mask','benefit_accuracy','benefit_majority_baseline','need_accuracy_cell_weighted','need_majority_baseline')} for s,v in summary['splits'].items()})))
