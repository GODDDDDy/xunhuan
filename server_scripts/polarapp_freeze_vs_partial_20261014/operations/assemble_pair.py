"""Minimal conversion of the existing verified continuation trainer/evaluator."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'run_training.py';s=p.read_text(encoding='utf-8')
start=s.index('def preflight():')
end=s.index('\ndef append_csv(',start)
preflight=r'''def preflight():
    """Real two-arm CUDA audit; all trial updates and random draws discarded."""
    config=json.loads((ROOT/'config/prepare.json').read_text())
    draft=json.loads((ROOT/'config/draft_run.json').read_text())
    rows=get_rows(config)
    schedules={n:json.loads((ROOT/'config'/f'{n}_schedule.json').read_text()) for n in ('joint','meta_inner','meta_outer')}
    datasets={'joint':dataset(rows,schedules['joint'])}
    for kind,key in (('meta_inner','meta_inner_scene_ids'),('meta_outer','meta_outer_scene_ids')):
        datasets[kind]=dataset([r for r in rows if r.scene in config[key]],schedules[kind])
    observations=[]
    for arm in ('frozen','partial'):
        parent=torch.load(ROOT/'weights/resume_parent_6000.pt',map_location='cpu',weights_only=False)
        up,down,fa,_,_=move_system()
        policy=configure(up,down,arm)
        opt,opt_fa=restore(up,down,fa,parent)
        fa.set_trainable_partition()
        frozen={'enhancer':state_hash(up.core),'task':fixed_task_hash(down),
                'FA_frozen':{n:p.detach().cpu().clone() for n,p in fa.named_parameters() if not p.requires_grad}}
        before_tail=tail_state(down)
        torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        inner=[to_device(datasets['meta_inner'][i]) for i in (1200,1201)]
        outer=[to_device(datasets['meta_outer'][i]) for i in (1200,1201)]
        meta_values=meta_update(up,down,fa,opt_fa,inner,outer,600)
        batches=[to_device(datasets['joint'][i]) for i in (12000,12001)]
        values=joint_update(up,down,fa,opt,batches,6001,draft['alignment_weight'])
        torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        check_frozen(up,down,fa,frozen)
        after_tail=tail_state(down)
        changed=sum(not torch.equal(before_tail[n],after_tail[n]) for n in before_tail)
        assert changed==0 if arm=='frozen' else changed>0
        assert meta_values['meta_hypergradient_scaled_norm']>0
        max_bytes=torch.cuda.max_memory_allocated()
        del inner,outer,batches,opt,opt_fa,before_tail,after_tail
        gc.collect();torch.cuda.empty_cache()
        from operations.pair_policy import load_tail
        load_tail(down,down._initial_tail)
        restored=restore(up,down,fa,parent);del restored
        from rgb_final_confirmation.data import load_manifest,PairedBranchDataset
        candidates=[r for r in load_manifest(config['manifest']) if r.split=='validation']
        def area(row):
            with Image.open(row.polar_000) as image:return image.width*image.height
        largest=max(candidates,key=area)
        batch=to_device(PairedBranchDataset([largest],'A1',None)[0])
        tick=time.perf_counter()
        with torch.no_grad():
            p,y=forward(up,down,batch,123,features=False)
            assert y.shape==batch['target'].shape and torch.isfinite(y).all()
        torch.cuda.synchronize()
        observations.append({'arm':arm,'policy':policy,'bilevel_joint_trial_seconds':elapsed,
            'peak_training_bytes':max_bytes,'meta_hypergradient_scaled_norm':meta_values['meta_hypergradient_scaled_norm'],
            'trial_changed_tail_tensors':changed,'frozen_audit_passed':True,
            'native_size':list(y.shape[-2:]),'native_forward_seconds':time.perf_counter()-tick,'trial_updates_discarded':True})
        del p,y,batch,up,down,fa,parent
        gc.collect();torch.cuda.empty_cache()
    budget=read_budget(ROOT)
    # Trials include a meta update on every timing observation: conservatively overestimate.
    projected_train=sum(x['bilevel_joint_trial_seconds'] for x in observations)*3000+2400
    projected_eval=max(x['native_forward_seconds'] for x in observations)*1060*3*1.5+1800
    available=budget['deadline_unix']-time.time()
    if projected_train+max(14400,projected_eval)>available:
        raise RuntimeError('Conservative paired training and full evaluation forecast exceeds immutable budget')
    draft['operational_training_deadline_unix']=min(budget['training_deadline_unix'],budget['deadline_unix']-max(14400,projected_eval))
    draft['budget']=budget
    draft['preflight_receipt']={'observations':observations,'projected_training_seconds':projected_train,
                              'projected_evaluation_seconds':projected_eval,'trial_updates_discarded':True}
    atomic_json(ROOT/'diagnostics/gpu_preflight.json',draft['preflight_receipt'])
    draft['code_sha256']={p.relative_to(ROOT).as_posix():sha(p)
        for folder in ('scripts','modules','source_snapshot','vendor','recovery','operations')
        for p in sorted((ROOT/folder).rglob('*.py')) if '__pycache__' not in p.parts}
    draft['protocol_id']=hashlib.sha256(json.dumps(draft,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    atomic_json(ROOT/'config/locked_run.json',draft)
    atomic_json(ROOT/'config/lock_receipt.json',{'protocol_id':draft['protocol_id'],'locked_sha256':sha(ROOT/'config/locked_run.json'),'locked_at_unix':time.time()})
    status('preflight_passed_protocol_locked',ROOT,protocol_id=draft['protocol_id'],observations=observations)
'''
s=s[:start]+preflight+'\n\n'+s[end:]
s=s.replace('"alignment_state":fa.state_dict() if arm=="meta" else None,','"alignment_state":fa.state_dict(),"task_tail":tail_state(down),')
s=s.replace('"optimizer_fa":optimizer_fa.state_dict() if arm=="meta" else None,','"optimizer_fa":optimizer_fa.state_dict(),')
s=s.replace('def fixed_validation_probe(up,down,locked,step):','def fixed_validation_probe(up,down,locked,step,arm):')
s=s.replace('if step==3000 and digest','if step==6000 and digest')
s=s.replace('"parent3000_PSNR"','"parent6000_PSNR"').replace('"parent3000_SSIM"','"parent6000_SSIM"')
s=s.replace('f"validation_probe_{step}.json"','f"{arm}_validation_probe_{step}.json"')
s=s.replace('"step":step,"records":records,"full_native_size_forward":True','"arm":arm,"step":step,"records":records,"full_native_size_forward":True')
s=s.replace('up,down,fa,receipt,config=move_system();fa.set_trainable_partition()',
            'up,down,fa,receipt,config=move_system();configure(up,down,arm);fa.set_trainable_partition()')
s=s.replace('"task":state_hash(down.task)','"task":fixed_task_hash(down)')
start=s.index('    optimizer=adapters_optimizer(up,down)\n',s.index('def train_arm'))
end=s.index('        log_path=',start)
s=s[:start]+'''    saved=verify_checkpoint(last,locked) if last.exists() else torch.load(ROOT/'weights/resume_parent_6000.pt',map_location='cpu',weights_only=False)
    start=saved['step'];meta_steps=saved['meta_steps']
    optimizer,optimizer_fa=restore(up,down,fa,saved)
    if last.exists():
'''+s[end:]
s=s.replace('if start==0:retention_probe(up,down,datasets["joint"],arm,0)',
            'if start==6000:retention_probe(up,down,datasets["joint"],arm,6000)')
s=s.replace('if start==3000 and not (ROOT/"downstream/metrics/validation_probe_3000.json").exists():\n        fixed_validation_probe(up,down,locked,3000)',
            'if start==6000 and not (ROOT/"downstream/metrics"/f"{arm}_validation_probe_6000.json").exists():\n        fixed_validation_probe(up,down,locked,6000,arm)')
s=s.replace('if arm=="meta" and step%10==0:', 'if step%10==0:')
s=s.replace('locked["alignment_weight"] if arm=="meta" else 0.', 'locked["alignment_weight"]')
s=s.replace('optimizer_fa.param_groups[0]["lr"] if arm=="meta" else 0.', 'optimizer_fa.param_groups[0]["lr"]')
s=s.replace('"lr_adapters":optimizer.param_groups[0]["lr"],',
            '"lr_adapters":optimizer.param_groups[0]["lr"],"lr_task_tail":optimizer.param_groups[1]["lr"] if len(optimizer.param_groups)>1 else 0.,')
s=s.replace('if step%1000==0:fixed_validation_probe(up,down,locked,step)',
            'if step%1000==0:fixed_validation_probe(up,down,locked,step,arm)')
s=s.replace('if locked["arms"] != ["meta"] or ARGS.arm not in ("meta","both"):',
            'if locked["arms"] != ["frozen","partial"] or ARGS.arm not in ("frozen","partial","both"):')
s=s.replace('User authorized meta-only training; cancelled control must not resume','Only paired frozen/partial collaboration is authorized')
p.write_text(s,encoding='utf-8')

p=root/'controller.py';s=p.read_text(encoding='utf-8')
s=s.replace('single-round ordinary-finetune and meta-collaboration, preflight included','new user-authorized paired frozen/partial collaboration; independent24h; parent not restarted')
s=s.replace('"--arm", "meta"','"--arm", "both"')
p.write_text(s,encoding='utf-8')

p=root/'run_evaluation.py';s=p.read_text(encoding='utf-8')
s=s.replace('NAMES = ("source", "meta")','NAMES = ("source", "frozen", "partial")')
s=s.replace('PAIRS = (("meta", "source"),)','PAIRS = (("frozen", "source"),("partial", "source"),("partial", "frozen"))')
s=s.replace('for _arm in ("meta",):','for _arm in ("frozen", "partial"):')
s=s.replace('for name in ("meta",):','for name in ("frozen", "partial"):')
s=s.replace('for arm in ("meta",):','for arm in ("frozen", "partial"):')
s=s.replace('if config["arms"] != ["meta"]:', 'if config["arms"] != ["frozen","partial"]:')
s=s.replace('joint_steps=3000','joint_steps=6000').replace('weights/resume_parent_3000.pt','weights/resume_parent_6000.pt')
s=s.replace('restored parent 3000-step adapters; not zero-adapter base','restored parent6000; two9000-step collaboration arms')
s=s.replace('3000-step parent float output','6000-step parent float output')
s=s.replace('"comparison_is_6000_vs_3000": True,','"comparison_is_frozen_vs_partial_same6000_start": True,')
s=s.replace('meta_minus_source_psnr=record["meta_minus_source_psnr"]','partial_minus_frozen_psnr=record["partial_minus_frozen_psnr"]')
s=s.replace('    del checkpoint\n    gc.collect()','    states["task_tail"] = {name:value.detach().clone() for name,value in checkpoint.get("task_tail",{}).items()}\n    del checkpoint\n    gc.collect()')
s=s.replace('    from modules.single_system import build_system','    from modules.single_system import build_system\n    from operations.pair_policy import tail_state,load_tail')
s=s.replace('    # Validate both strict adapter schemas before any final data forward.',
            '    initial_tail=tail_state(down)\n    # Validate strict adapter/tail schemas before any final data forward.')
s=s.replace('    started = time.time()','    for name in NAMES:\n        load_tail(down,states[name]["task_tail"] or initial_tail)\n    started = time.time()')
s=s.replace('                p_source, source_features = up(','                load_tail(down,initial_tail)\n                p_source, source_features = up(')
s=s.replace('                for name in NAMES:\n                    guard(','                for name in NAMES:\n                    load_tail(down,states[name]["task_tail"] or initial_tail)\n                    guard(')
s=s.replace('"matched_finetuning_control_available": False,','"matched_finetuning_control_available": False, "matched_freeze_policy_collaboration_available": True,')
p.write_text(s,encoding='utf-8')

p=root/'archive.py';s=p.read_text(encoding='utf-8')
s=s.replace('locked.get("arms") != ["meta"]','locked.get("arms") != ["frozen","partial"]')
s=s.replace('current user-authorized scope requires meta-only training','User authorization requires paired frozen/partial collaboration')
s=s.replace('for arm in ("source", "meta"):', 'for arm in ("source", "frozen", "partial"):')
s=s.replace('for pair in ("meta_minus_source",):','for pair in ("frozen_minus_source", "partial_minus_source", "partial_minus_frozen"):')
s=s.replace('for arm in ("meta",):','for arm in ("frozen", "partial"):')
s=s.replace('required_sources["resume_parent_3000"]','required_sources["resume_parent_6000"]')
p.write_text(s,encoding='utf-8')
for p in list(root.glob('*.py'))+list((root/'operations').glob('*.py')):
    compile(p.read_text(encoding='utf-8'),str(p),'exec')
print('Paired code assembled and syntax checked.')
