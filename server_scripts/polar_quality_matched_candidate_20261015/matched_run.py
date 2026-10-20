"""One paired short experiment. All old runs are read-only."""
import argparse,copy,csv,gc,hashlib,json,time,shutil,sys
from pathlib import Path
import numpy as np
import torch
from common import ROOT,seed,sha,save,json_save,rng_state,restore_rng,guard_gpu,status
import gain22,staged,study
from quality_forward import shared_first,candidate_from_first

def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def load(device):
    cfg=json.loads((ROOT/'config/execution_config.json').read_text());plan=json.loads((ROOT/'config/plan.json').read_text())
    assert sha(ROOT/'config/execution_config.json')==plan['execution_config_sha256']
    for x in plan['code_manifest']:assert sha(ROOT/x['path'])==x['sha256'],x['path']
    assert sha(ROOT/'weights/common_initial21.pt')==plan['source_initial_sha256']
    seed(plan['seed']);model,task,raw=gain22._components(cfg,device)
    source=torch.load(ROOT/'weights/common_initial21.pt',map_location='cpu',weights_only=False)
    assert source['protocol_id']==plan['source_initial_protocol']
    model.load_state_dict(source['r2p_and_projections'],strict=True);task.net_g.load_state_dict(source['tasknet_net_g'],strict=True)
    for k in ('tau','cost_scale','cost_bias','safety_margin'):setattr(model,k,source[k])
    model.step=int(source['step']);model.eval();task.eval();model.requires_grad_(False);task.requires_grad_(False)
    audit(model,task,source,initial=True)
    imported=[]
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file:
            path=Path(file).resolve()
            if path.suffix=='.py' and path.is_relative_to(ROOT.parent) and not path.is_relative_to(ROOT):
                value=sha(path);target=ROOT/'runtime_source'/value[:12]/path.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
                imported.append(dict(original=str(path),sha256=value,archived=str(target.relative_to(ROOT))))
    json_save(ROOT/'diagnostics/runtime_source_manifest.json',dict(files=imported,torch_version=torch.__version__,numpy_version=np.__version__,python=sys.version))
    json_save(ROOT/'diagnostics/source_initial_load.json',dict(actual_source_sha256=plan['source_initial_sha256'],actual_source_protocol=source['protocol_id'],source_model_tensors=len(source['r2p_and_projections']),net_g_tensors=len(source['tasknet_net_g']),strict_equal=True,quality_training_history=True,optimizer_reused=False,official_weights_only_genealogy=True))
    return cfg,plan,model,task,raw,source
def audit(model,task,source,initial=False):
    for k,v in model.state_dict().items():
        if initial or not k.startswith(('feedback.','error_guidance.')):assert torch.equal(v.cpu(),source['r2p_and_projections'][k]),k
    assert all(torch.equal(v.cpu(),source['tasknet_net_g'][k]) for k,v in task.net_g.state_dict().items())
def configure(model,task):
    model.eval();task.eval();model.requires_grad_(False);task.requires_grad_(False)
    model.feedback.requires_grad_(True).train();model.error_guidance.requires_grad_(True).train()
def forward(model,task,batch,arm,b0,capture=False):
    first=shared_first(model,task,batch,study_module=study)
    return candidate_from_first(model,task,first,arm,fixed_b0=b0,capture_paths=capture,study_module=study)
def checkpoint(model,task,opt,step,arm,plan):
    return dict(r2p_and_projections=model.state_dict(),tasknet_net_g=task.net_g.state_dict(),tau=model.tau,cost_scale=model.cost_scale,cost_bias=model.cost_bias,safety_margin=model.safety_margin,
                step=step,arm=arm,protocol_id=plan['protocol_id'],optimizer=opt.state_dict(),rng=rng_state(),source_initial_sha256=plan['source_initial_sha256'])
@torch.no_grad()
def calibration(cfg,model,task,raw,device):
    total=0.;n=0;records=[]
    for i,rawbatch in enumerate(staged.loader(cfg,raw,'judgement',8)):
        for micro in study.microbatches(rawbatch):
            if device.type=='cuda':guard_gpu()
            batch=study.base.unpack_batch(micro,device);study.base.set_step_seed(cfg['seed'],'quality_constant_calibration',i,batch['sample_id'])
            first=shared_first(model,task,batch,study_module=study);total+=float(first.error_scale.double().sum());n+=first.error_scale.numel()
            assert batch['sample_id'][0].startswith('train/')
            records.append(dict(sample_id=batch['sample_id'][0],scene=batch['scene'][0],b_mean=float(first.error_scale.mean())))
    assert len(records)==16
    b0=total/n;json_save(ROOT/'config/fixed_calibration.json',dict(b0=b0,c0=float(np.exp(-b0/model.tau)),tau=model.tau,crops=records,validation_used=False,test_used=False));return b0
def cpu_verify():
    torch.set_num_threads(2);cfg,plan,model,task,raw,source=load(torch.device('cpu'))
    small=copy.deepcopy(cfg);small['data']['crop_size']=32
    batch=study.base.unpack_batch(next(study.microbatches(next(iter(staged.loader(small,raw,'candidate',1))))),torch.device('cpu'))
    seed(100)
    with torch.no_grad():old=study.recurrent_forward(model,task,batch)
    seed(100)
    with torch.no_grad():out=forward(model,task,batch,'real_quality',.03,True)
    assert torch.equal(old.t1,out.t1) and torch.equal(old.ungated_t2,out.ungated_t2)
    assert out.path_audit['all_quality_paths_match'];t1=digest(out.t1);u=digest(out.ungated_t2);del old,out
    altered=dict(batch,target=1-batch['target']);seed(100)
    with torch.no_grad():out=forward(model,task,altered,'real_quality',.03)
    assert digest(out.ungated_t2)==u;del out
    configure(model,task);opt=staged.phase_optimizer(model,'candidate');snapshot=copy.deepcopy(checkpoint(model,task,opt,0,'real_quality',plan))
    def update():
        seed(100);opt.zero_grad(set_to_none=True);out=forward(model,task,batch,'real_quality',.03,False);assert digest(out.t1)==t1
        qgrad=torch.autograd.grad((out.ungated_t2-batch['target']).square().mean(),tuple(model.error_guidance.predict.parameters()),allow_unused=True,retain_graph=True)
        assert all(g is None or torch.count_nonzero(g)==0 for g in qgrad),'Candidate loss must not modify Q predictor'
        loss=staged.candidate_loss(out,batch['target']);loss.backward()
        assert all(p.grad is None for k,p in model.named_parameters() if not k.startswith(('feedback.','error_guidance.')))
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.feedback.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.error_guidance.predict.parameters())
        torch.nn.utils.clip_grad_norm_(staged.phase_module(model,'candidate').parameters(),1,error_if_nonfinite=True);opt.step();audit(model,task,source)
        return {k:v.detach().clone() for k,v in model.state_dict().items()},float(loss)
    expected,l1=update();model.load_state_dict(snapshot['r2p_and_projections']);task.net_g.load_state_dict(snapshot['tasknet_net_g']);opt.load_state_dict(snapshot['optimizer']);restore_rng(snapshot['rng']);actual,l2=update()
    assert l1==l2 and all(torch.equal(v,actual[k]) for k,v in expected.items())
    model.load_state_dict(source['r2p_and_projections']);model.eval();task.eval();seed(100)
    with torch.no_grad():fixed=forward(model,task,batch,'fixed_quality',.03,True)
    assert digest(fixed.t1)==t1 and fixed.path_audit['all_quality_paths_match']
    json_save(ROOT/'diagnostics/cpu_checks.json',dict(passed=True,strict_source_tensors=len(source['r2p_and_projections']),net_g_tensors=len(source['tasknet_net_g']),source_initial_sha256=plan['source_initial_sha256'],real_forward_exact=True,GT_invariance=True,optimizer_and_rng_resume_exact=True,gradients_only_feedback_Q=True,Q_prediction_supervision_gradient=True,first_round_and_core_frozen=True,all_condition_paths_match=True,diagnostic_updates_discarded=True))
    status('cpu_verified')
def preflight():
    torch.set_num_threads(2);cfg,plan,model,task,raw,source=load(torch.device('cuda:0'));guard_gpu();b0=calibration(cfg,model,task,raw,torch.device('cuda:0'))
    configure(model,task);opt=staged.phase_optimizer(model,'candidate');times=[]
    for i,rb in enumerate(staged.loader(cfg,raw,'candidate',4)):
        guard_gpu();tick=time.time();opt.zero_grad(set_to_none=True)
        for micro in study.microbatches(rb):
            batch=study.base.unpack_batch(micro,torch.device('cuda:0'));study.base.set_step_seed(cfg['seed'],'preflight_candidate',i,batch['sample_id']);out=forward(model,task,batch,'real_quality',b0);loss=staged.candidate_loss(out,batch['target']);(loss/2).backward();del out
        torch.nn.utils.clip_grad_norm_(staged.phase_module(model,'candidate').parameters(),1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();times.append(time.time()-tick)
    audit(model,task,source);del opt,model,task;gc.collect();torch.cuda.empty_cache()
    cfg,plan,model,task,raw,source=load(torch.device('cuda:0'))
    # Largest known original validation image; no test outcomes inspected.
    from rgb_final_confirmation.data import load_manifest,select_rows,PairedBranchDataset
    from PIL import Image
    rows=select_rows(load_manifest(cfg['paths']['manifest']),'validation');sizes=[]
    for i,r in enumerate(rows):
        with Image.open(r.official_rgb) as im:sizes.append((im.width*im.height,i))
    sample=PairedBranchDataset(rows,group='A1',crop_size=None)[max(sizes)[1]];batch={k:v[None].cuda() if torch.is_tensor(v) else [v] for k,v in sample.items()}
    guard_gpu();seed(2026);tick=time.time()
    with torch.no_grad():
        first=shared_first(model,task,batch,study_module=study)
        for arm in plan['arms']:
            out=candidate_from_first(model,task,first,arm,fixed_b0=b0,capture_paths=True,study_module=study);assert out.path_audit['all_quality_paths_match'];del out
    torch.cuda.synchronize();elapsed=time.time()-tick
    estimate=1.5*(4000*float(np.median(times[1:]))+1060*elapsed)+1800
    report=dict(step_seconds=float(np.median(times[1:])),paired_full_frame_seconds=elapsed,full_frame_shape=list(batch['polar'].shape),peak_vram_mib=torch.cuda.max_memory_allocated()/2**20,estimated_total_seconds=estimate,resource_rule_passed=estimate<plan['budget_seconds'],diagnostic_updates_discarded=True,original_size=True,tiling=False)
    json_save(ROOT/'diagnostics/benchmark.json',report);assert report['resource_rule_passed'],report
    locked=dict(plan,benchmark=report,fixed_calibration_sha256=sha(ROOT/'config/fixed_calibration.json'));json_save(ROOT/'config/locked_run.json',locked);json_save(ROOT/'config/lock_receipt.json',dict(protocol_id=plan['protocol_id'],sha256=sha(ROOT/'config/locked_run.json'),preflight_updates_discarded=True));status('locked_ready_to_train')
@torch.no_grad()
def monitor(model,task,cfg,raw,device,arm,b0,step):
    before=rng_state();model.eval();task.eval();rows=[]
    for i,rb in enumerate(staged.loader(cfg,raw,'bias',2)):
        for micro in study.microbatches(rb):
            guard_gpu();batch=study.base.unpack_batch(micro,device);study.base.set_step_seed(cfg['seed'],'fixed_train_monitor',i,batch['sample_id']);out=forward(model,task,batch,arm,b0)
            rows.append(dict(sample_id=batch['sample_id'][0],t1_sha256=digest(out.t1),candidate_mse=float((out.ungated_t2-batch['target']).square().mean())))
    directory=ROOT/'arms'/arm/'monitor'
    if step>0:
        previous=json.loads((directory/'step_0.json').read_text())['rows'];assert [(r['sample_id'],r['t1_sha256']) for r in rows]==[(r['sample_id'],r['t1_sha256']) for r in previous]
    json_save(directory/f'step_{step}.json',dict(step=step,training_only=True,selection=False,first_round_fixed=True,rows=rows));restore_rng(before)
def train(arm):
    device=torch.device('cuda:0');torch.set_num_threads(2);cfg,plan,model,task,raw,source=load(device);b0=json.loads((ROOT/'config/fixed_calibration.json').read_text())['b0']
    opt=staged.phase_optimizer(model,'candidate');directory=ROOT/'arms'/arm;start=0
    if (directory/'last.pt').exists():
        ck=torch.load(directory/'last.pt',map_location='cpu',weights_only=False);assert ck['protocol_id']==plan['protocol_id'] and ck['arm']==arm
        model.load_state_dict(ck['r2p_and_projections']);task.net_g.load_state_dict(ck['tasknet_net_g']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng']);start=ck['step']
    monitor(model,task,cfg,raw,device,arm,b0,start);budget=json.loads((ROOT/'budget.json').read_text());deadline=budget['train_deadline_unix'];seen=set();step=start
    for rb in staged.loader(cfg,raw,'candidate',2000,start):
        guard_gpu();assert time.time()<deadline,'Training deadline reached; preserve last endpoint, incomplete'
        step+=1;configure(model,task);opt.zero_grad(set_to_none=True);tick=time.time();ids=[];lossval=0.
        for micro in study.microbatches(rb):
            batch=study.base.unpack_batch(micro,device);ids+=batch['sample_id'];study.base.set_step_seed(cfg['seed'],'training_candidate',step,batch['sample_id'])
            out=forward(model,task,batch,arm,b0);loss=staged.candidate_loss(out,batch['target']);assert torch.isfinite(loss);(loss/2).backward();lossval+=float(loss)/2;del out
        torch.nn.utils.clip_grad_norm_(staged.phase_module(model,'candidate').parameters(),1,error_if_nonfinite=True);opt.step();seen.update(ids)
        record=dict(arm=arm,step=step,loss=lossval,seconds=time.time()-tick,sample_ids=';'.join(ids),feedback_lr=1e-5,Q_lr=1e-4,core_lr=0.,task_lr=0.)
        with (directory/'train.csv').open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=record)
            if f.tell()==0:w.writeheader()
            w.writerow(record)
        if step%100==0:status('training',arm=arm,step=step,target=2000)
        if step%500==0:
            audit(model,task,source);save(directory/'last.pt',checkpoint(model,task,opt,step,arm,plan));save(directory/f'step_{step}.pt',checkpoint(model,task,opt,step,arm,plan));monitor(model,task,cfg,raw,device,arm,b0,step)
    assert step==2000
    allrows=list(csv.DictReader((directory/'train.csv').open()));used={sid for r in allrows for sid in r['sample_ids'].split(';')}
    from rgb_final_confirmation.data import load_manifest,select_rows
    expected={r.sample_id for r in select_rows(load_manifest(cfg['paths']['manifest']),'train') if r.scene in cfg['staged']['scene_roles']['candidate']}
    assert len(expected)==3640 and expected==used,(len(expected),len(used))
    save(directory/'final.pt',checkpoint(model,task,opt,step,arm,plan));json_save(directory/'selection.json',dict(step=step,sha256=sha(directory/'final.pt'),rule='fixed2000 endpoint',test_used=False,unique_training_images=len(used),candidate_role_all_covered=True));status('arm_training_complete',arm=arm,step=step)
@torch.no_grad()
def evaluate(split):
    device=torch.device('cuda:0');torch.set_num_threads(2);cfg,plan,model,task,raw,source=load(device);b0=json.loads((ROOT/'config/fixed_calibration.json').read_text())['b0']
    states={a:torch.load(ROOT/'arms'/a/'final.pt',map_location='cpu',weights_only=False) for a in plan['arms']};outputs=[];examples=set()
    for a,ck in states.items():
        assert ck['step']==2000 and ck['arm']==a and ck['protocol_id']==plan['protocol_id']
        for k,v in source['r2p_and_projections'].items():
            if not k.startswith(('feedback.','error_guidance.')):assert torch.equal(v,ck['r2p_and_projections'][k]),k
        assert all(torch.equal(v,ck['tasknet_net_g'][k]) for k,v in source['tasknet_net_g'].items())
    from rgb_final_confirmation.evaluation import psnr,ssim
    for i,rb in enumerate(study.base.split_loader(cfg,split)):
        guard_gpu();batch=study.base.unpack_batch(rb,device);study.base.set_step_seed(cfg['seed'],'matched_eval_'+split,i,batch['sample_id']);model.eval();task.eval()
        # Only frozen tensors enter first; assert against common initial below.
        first=shared_first(model,task,batch,study_module=study);record=dict(sample_id=batch['sample_id'][0],scene=batch['scene'][0],t1_sha256=digest(first.t1),t1=dict(psnr=psnr(first.t1,batch['target']),ssim=ssim(first.t1,batch['target'])),arms={})
        arrays=dict(image=batch['official_rgb'][0].cpu().numpy(),gt=batch['target'][0].cpu().numpy(),t1=first.t1[0].cpu().numpy()) if record['scene'] not in examples else None
        for a in plan['arms']:
            for prefix,module in (('feedback.',model.feedback),('error_guidance.',model.error_guidance)):
                module.load_state_dict({k.removeprefix(prefix):v for k,v in states[a]['r2p_and_projections'].items() if k.startswith(prefix)},strict=True)
            model.eval()
            out=candidate_from_first(model,task,first,a,fixed_b0=b0,capture_paths=True,study_module=study);assert out.path_audit['all_quality_paths_match']
            record['arms'][a]=dict(psnr=psnr(out.ungated_t2,batch['target']),ssim=ssim(out.ungated_t2,batch['target']),u_sha256=digest(out.ungated_t2),t1_equal=torch.equal(out.t1,first.t1))
            if arrays is not None:arrays[a]=out.ungated_t2[0].cpu().numpy()
            del out
        if arrays is not None:
            d=ROOT/'examples'/split;d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/(record['sample_id'].replace('/','_')+'.npz'),**arrays);examples.add(record['scene'])
        outputs.append(record);json_save(ROOT/'metrics'/f'{split}_partial.json',dict(rows=outputs,original_size=True,tiling=False))
        if (i+1)%10==0:status('evaluating',split=split,completed=i+1,total=872 if split=='validation' else 188)
        del first,batch;gc.collect();torch.cuda.empty_cache()
    assert len(outputs)==(872 if split=='validation' else 188)
    scenes=[]
    for scene in sorted({r['scene'] for r in outputs}):
        members=[r for r in outputs if r['scene']==scene];means={name:{m:float(np.mean([(r['t1'] if name=='t1' else r['arms'][name])[m] for r in members])) for m in ('psnr','ssim')} for name in ['t1']+plan['arms']};scenes.append(dict(scene=scene,frames=len(members),means=means))
    summary={name:{m:float(np.mean([(r['t1'] if name=='t1' else r['arms'][name])[m] for r in outputs])) for m in ('psnr','ssim')} for name in ['t1']+plan['arms']}
    json_save(ROOT/'metrics'/f'final_{split}.json',dict(rows=outputs,per_scene=scenes,image_means=summary,original_size=True,tiling=False,protocol_id=plan['protocol_id'],selection=False))
    status('split_complete',split=split,frames=len(outputs))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','preflight','train','evaluate']);p.add_argument('--arm');p.add_argument('--split');a=p.parse_args()
    if a.mode=='cpu':cpu_verify()
    elif a.mode=='preflight':preflight()
    elif a.mode=='train':train(a.arm)
    else:evaluate(a.split)
