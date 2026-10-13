import csv,hashlib,json,math,shutil
from common import ROOT,json_save,sha

def prepare():
    assert json.loads((ROOT/'status.json').read_text())['stage']=='diagnostic_completed'
    cfg=json.loads((ROOT/'config/diagnostic.json').read_text());cfg.pop('diagnostic',None)
    with open(cfg['paths']['manifest']) as f:rows=[r for r in csv.DictReader(f) if r['split']=='train']
    assert len(rows)==5440
    scenes=sorted({r['scene'] for r in rows},key=lambda x:hashlib.sha256(f'2026:staged_scene:{x}'.encode()).hexdigest())
    assert len(scenes)==51
    roles=dict(candidate=scenes[:31],judgement=scenes[31:43],bias=scenes[43:47],safety=scenes[47:])
    original=json.loads(open(cfg['paths']['schedule']).read());schedules={};counts={};coverage={}
    for role,selected in roles.items():
        indices={i for i,r in enumerate(rows) if r['scene'] in selected};pool=[x for x in original['items'] if x['sample_index'] in indices]
        first={}
        for item in pool:first.setdefault(item['sample_index'],item)
        assert set(first)==indices
        base=list(first.values());count=6000 if role in ('candidate','judgement') else math.ceil(len(base)/2)*2
        assert count>=len(base)
        extra=[pool[i%len(pool)] for i in range(count-len(base))];items=base+extra
        payload=dict(schema_version=1,seed=2026,total_steps=count//2,batch_size=2,crop_size=128,items=items)
        payload['sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        path=ROOT/'config'/f'{role}_schedule.json';json_save(path,payload);schedules[role]=str(path);counts[role]=len(indices);coverage[role]=[rows[i]['sample_id'] for i in sorted(indices)]
    assert set().union(*(set(ids) for ids in coverage.values()))=={r['sample_id'] for r in rows}
    cfg['staged']=dict(steps=dict(candidate=3000,judgement=3000,bias=json.loads(open(schedules['bias']).read())['total_steps'],safety=json.loads(open(schedules['safety']).read())['total_steps']),minimum_steps=dict(candidate=max(1500,math.ceil(counts['candidate']/2)),judgement=max(1500,math.ceil(counts['judgement']/2))),schedules=schedules,scene_roles=roles,sample_counts=counts,safety_margin_grid=[0.,.05,.1,.2,.4,.8,1.6,3.2,6.4,12.8],candidate_loss='RGB MSE + .02 edge MAE + .1 positive local MSE difference',judge_loss='SmoothL1 normalized action costs + .1 expected true normalized cost',action_cost_scale='judgement-training 256 crops absolute differences95% percentile floor1e-5',fixed_first_round=True,reuse_optimizer=False,reuse_rng=False,steps_selected_from_validation=False)
    cfg['optimization']=dict(candidate_lr=1e-5,action_judgement_lr=1e-4,core_lr=0.,task_lr=0.)
    cfg['protocol_id']='preflight'
    json_save(ROOT/'config/staged_preflight.json',cfg)
    json_save(ROOT/'config/training_role_manifest.json',dict(total_train_images=5440,scene_roles=roles,sample_counts=counts,sample_ids=coverage,rule='SHA256(2026:staged_scene:scene);31/12/4/4 scenes',test_used=False,validation_used=False,inherited_model_previously_saw_these_train_scenes=True))
    snapshot=ROOT/'diagnostics/candidate_diagnostic_snapshot';snapshot.mkdir()
    for name in ('study.py','candidate_diagnostic.py','diagnostic_controller.py'):
        shutil.copy2(ROOT/'scripts'/name,snapshot/name)
    p=ROOT/'scripts/study.py';text=p.read_text();needle='second.ungated_t2=ungated_t2;second.acceptance=judgement;second.update_mask=update_mask'
    assert needle in text
    text=text.replace(needle,needle+"\n    second.judgement_feature=f2['T1'];second.judgement_state=context['state']['level1']")
    p.write_text(text)
    json_save(ROOT/'diagnostics/staged_decision.json',dict(fixed_validation_frames=28,source_sha256=cfg['reuse']['sha256'],oracle_measured_gain=0.123685093391137,old_gate_gain=0.00034641343334,decision='Both candidate and judgement require work; execute predeclared candidate->fixed-candidate action-cost judgement sequence',test_used=False,diagnostic_reference_not_deployable=True))
    print(json.dumps(dict(prepared=True,roles=counts)))

if __name__=='__main__':prepare()
