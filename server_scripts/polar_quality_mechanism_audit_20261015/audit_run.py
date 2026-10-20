"""Read-only Experiment27 baseline completion; no training entry point."""
import argparse, gc, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from common import ROOT, guard_gpu, json_save, seed, sha, status
import gain22, staged, study
from quality_forward import shared_first, candidate_from_first

def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def load(device):
    plan=json.loads((ROOT/'config/locked_run.json').read_text())
    for item in plan['code_manifest']:
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
    cfg=json.loads((ROOT/'config/execution_config.json').read_text())
    assert sha(ROOT/'config/execution_config.json')==plan['execution_config_sha256']
    assert sha(ROOT/'weights/common_initial21.pt')==plan['source_initial_sha256']
    seed(2026);model,task,raw=gain22._components(cfg,device)
    source=torch.load(ROOT/'weights/common_initial21.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(source['r2p_and_projections'],strict=True)
    task.net_g.load_state_dict(source['tasknet_net_g'],strict=True)
    for name in ('tau','cost_scale','cost_bias','safety_margin'):setattr(model,name,source[name])
    model.eval().requires_grad_(False);task.eval().requires_grad_(False)
    assert all(torch.equal(v.cpu(),source['r2p_and_projections'][k]) for k,v in model.state_dict().items())
    assert all(torch.equal(v.cpu(),source['tasknet_net_g'][k]) for k,v in task.net_g.state_dict().items())
    evidence=[]
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file:
            p=Path(file).resolve()
            if p.suffix=='.py' and p.is_relative_to(ROOT.parent) and not p.is_relative_to(ROOT):
                value=sha(p);target=ROOT/'runtime_source'/value[:12]/p.name
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
                evidence.append(dict(original=str(p),archived=str(target.relative_to(ROOT)),sha256=value))
    json_save(ROOT/'diagnostics/runtime_source_manifest.json',dict(files=evidence))
    return cfg,plan,model,task,raw,source

@torch.no_grad()
def cpu():
    torch.set_num_threads(2);cfg,plan,model,task,raw,source=load(torch.device('cpu'))
    small=json.loads(json.dumps(cfg));small['data'].update(crop_size=32,workers=0)
    batch=study.base.unpack_batch(next(study.microbatches(next(iter(staged.loader(small,raw,'candidate',1))))),torch.device('cpu'))
    b0=json.loads((ROOT/'config/fixed_calibration.json').read_text())['b0']
    counts={'E':0,'D':0,'M':0};hooks=[]
    for module,key in [(model.r2p,'E'),(task.net_g,'D'),(model.action_gate,'M'),(model.update_gate,'M')]:
        hooks.append(module.register_forward_hook(lambda *args,key=key: counts.__setitem__(key,counts[key]+1)))
    seed(100);first=shared_first(model,task,batch,study_module=study)
    out=candidate_from_first(model,task,first,'real_quality',b0,capture_paths=True,study_module=study)
    assert counts=={'E':2,'D':2,'M':0},counts
    t1=digest(out.t1);u=digest(out.ungated_t2);paths=out.path_audit
    for h in hooks:h.remove()
    altered=dict(batch,target=1-batch['target']);seed(100)
    other=shared_first(model,task,altered,study_module=study)
    changed=candidate_from_first(model,task,other,'real_quality',b0,capture_paths=True,study_module=study)
    assert digest(changed.t1)==t1 and digest(changed.ungated_t2)==u
    assert paths['all_quality_paths_match'] and changed.path_audit['all_quality_paths_match']
    assert all(torch.equal(v.cpu(),source['r2p_and_projections'][k]) for k,v in model.state_dict().items())
    json_save(ROOT/'diagnostics/cpu_checks.json',dict(passed=True,actual_calls=counts,GT_invariant=True,
        model_tensors=len(source['r2p_and_projections']),net_g_tensors=len(source['tasknet_net_g']),
        strict_initial_equal=True,all_parameters_frozen=True,all_quality_paths_match=True,training=False))
    status('cpu_verified')

@torch.no_grad()
def evaluate():
    torch.set_num_threads(2);guard_gpu()
    cfg,plan,model,task,raw,source=load(torch.device('cuda:0'));del source;gc.collect();torch.cuda.empty_cache()
    b0=json.loads((ROOT/'config/fixed_calibration.json').read_text())['b0']
    reference=json.loads((ROOT/'config/source27_final_validation.json').read_text())
    expected={r['sample_id']:r for r in reference['rows']};assert len(expected)==872
    from rgb_final_confirmation.evaluation import psnr,ssim
    partial=ROOT/'metrics/baseline_partial.json'
    outputs=json.loads(partial.read_text())['rows'] if partial.exists() else []
    already={r['sample_id'] for r in outputs};examples={r['scene'] for r in outputs}
    first_times=[]
    for i,rb in enumerate(study.base.split_loader(cfg,'validation')):
        sid=rb['sample_id'][0]
        if sid in already:continue
        guard_gpu();tick=time.time();batch=study.base.unpack_batch(rb,torch.device('cuda:0'))
        study.base.set_step_seed(2026,'matched_eval_validation',i,batch['sample_id'])
        first=shared_first(model,task,batch,study_module=study)
        ref=expected[sid];assert ref['scene']==batch['scene'][0]
        assert digest(first.t1)==ref['t1_sha256'],('T1 must exactly match Experiment27',sid)
        record=dict(sample_id=sid,scene=ref['scene'],t1_sha256=digest(first.t1),
            t1=dict(psnr=psnr(first.t1,batch['target']),ssim=ssim(first.t1,batch['target'])),arms={})
        assert all(abs(record['t1'][m]-ref['t1'][m])<1e-10 for m in ('psnr','ssim'))
        arrays=None
        if record['scene'] not in examples:
            arrays={k:v[0].cpu().numpy() for k,v in dict(image=first.image,gt=batch['target'],t1=first.t1,
                b=first.error_scale,c=first.confidence_map).items()}
        for arm in ('real_quality','fixed_quality'):
            guard_gpu();out=candidate_from_first(model,task,first,arm,b0,capture_paths=True,study_module=study)
            assert out.path_audit['all_quality_paths_match']
            record['arms'][arm]=dict(psnr=psnr(out.ungated_t2,batch['target']),ssim=ssim(out.ungated_t2,batch['target']),
                u_sha256=digest(out.ungated_t2),t1_equal=torch.equal(out.t1,first.t1))
            if arrays is not None:arrays[arm]=out.ungated_t2[0].cpu().numpy()
            if i==0:json_save(ROOT/'diagnostics'/f'{arm}_actual_paths.json',out.path_audit)
            del out
        if arrays is not None:
            directory=ROOT/'examples/validation';directory.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(directory/(sid.replace('/','_')+'.npz'),**arrays);examples.add(record['scene'])
        outputs.append(record)
        json_save(partial,dict(rows=outputs,original_size=True,tiling=False,training=False,test_used=False))
        if (i+1)%10==0:status('evaluating_baseline',completed=len(outputs),total=872)
        torch.cuda.synchronize();first_times.append(time.time()-tick)
        if i==2:
            estimate=1.1*(872-3)*float(np.median(first_times[1:]))+300
            budget=json.loads((ROOT/'budget.json').read_text())
            json_save(ROOT/'diagnostics/first_frame_resource.json',dict(paired_seconds=first_times[-1],
                estimated_remaining_seconds=estimate,deadline_unix=budget['deadline_unix'],original_size=True,
                t1_exact_source27=True,training=False))
            assert time.time()+estimate<budget['deadline_unix'],'Original-size completion cannot fit fixed3h budget'
        del first,batch;gc.collect();torch.cuda.empty_cache()
    assert len(outputs)==872 and len({r['sample_id'] for r in outputs})==872
    json_save(ROOT/'metrics/baseline_validation.json',dict(rows=outputs,original_size=True,tiling=False,
        training=False,test_used=False,protocol_id=plan['protocol_id'],all_T1_hashes_equal_source27=True))
    status('inference_completed',frames=872,training=False,test_used=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['cpu','evaluate'])
    mode=parser.parse_args().mode
    cpu() if mode=='cpu' else evaluate()
