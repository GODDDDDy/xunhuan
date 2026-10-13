"""Minimal fixed-candidate predictability audit and one conditional judgement fit."""
import copy,csv,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import ROOT,json_save,save,seed,sha,rng_state,restore_rng,guard_gpu,status
import staged,study
from action_gate import ActionCostGate,ACTIONS,action_costs,cost_loss
from acceptance import resize
from quality_metrics import spearman_numpy
OUT=ROOT/'downstream'
_components=staged.components
def correlation(a,b):
    if np.std(a)<1e-12 or np.std(b)<1e-12:return None
    value=float(spearman_numpy(a,b));return value if np.isfinite(value) else None

class FloatActionCostGate(ActionCostGate):
    """Keep the source body; preserve floats and expose within-cell delta energy."""
    def __init__(self,source):
        nn.Module.__init__(self)
        self.feature_projection=copy.deepcopy(source.feature_projection)
        self.body=copy.deepcopy(source.body)
        old=self.body[0]
        new=nn.Conv2d(old.in_channels+7,old.out_channels,3,padding=1).to(old.weight.device)
        with torch.no_grad():
            new.weight.zero_();new.weight[:,:old.in_channels].copy_(old.weight);new.bias.copy_(old.bias)
        self.body[0]=new
    def inputs(self,image,polar,t1,proposal,error_scale,confidence,feature,state):
        a,p=t1.detach().float(),proposal.detach().float();d=p-a
        maps=[F.avg_pool2d(x.detach().float(),4) for x in (image,a,p,d,polar)]
        size=maps[0].shape[-2:]
        maps += [resize(error_scale.detach(),size),resize(confidence.detach(),size),self.feature_projection(resize(feature.detach(),size)),resize(state.detach(),size)]
        mean=F.avg_pool2d(d,4);absolute=F.avg_pool2d(d.abs(),4)
        maps += [F.avg_pool2d(d.square(),4),absolute,(mean.abs().sum(1,keepdim=True)/(absolute.sum(1,keepdim=True)+1e-8))]
        return torch.cat(maps,1)
    def forward(self,image,polar,t1,proposal,error_scale,confidence,feature,state,bias=None,margin=0.):
        raw=self.body(self.inputs(image,polar,t1,proposal,error_scale,confidence,feature,state))
        corrected=raw if bias is None else raw+raw.new_tensor(bias)[None,:,None,None]
        costs=torch.cat((torch.zeros_like(corrected[:,:1]),corrected+margin),1)
        decision=costs.argmin(1,keepdim=True);alpha=raw.new_tensor(ACTIONS)[decision]
        return dict(raw_cost=raw,corrected_cost=corrected,costs=costs,decision=decision,mask=F.interpolate(alpha,size=image.shape[-2:],mode='nearest'))

def components(cfg,device):
    model,task,raw=_components(cfg,device)
    model.action_gate=FloatActionCostGate(model.action_gate).to(device)
    model.requires_grad_(False);task.requires_grad_(False);model.eval();task.eval()
    return model,task,raw
staged.components=components

def args(out,batch):
    return (batch['official_rgb'],study.polarization_summary(batch['polar']),out.t1,out.ungated_t2,out.error_scale,out.confidence_map,out.judgement_feature,out.judgement_state)

def source_gate(model,cfg,device):
    gate=ActionCostGate(model.update_gate).to(device)
    source=torch.load(cfg['reuse']['checkpoint'],map_location='cpu',weights_only=False)
    gate.load_state_dict({k.removeprefix('action_gate.'):v for k,v in source['r2p_and_projections'].items() if k.startswith('action_gate.')},strict=True)
    gate.requires_grad_(False).eval();return gate

def frozen_audit(model,task,cfg):
    source=torch.load(cfg['reuse']['checkpoint'],map_location='cpu',weights_only=False)
    frozen={n:v for n,v in model.state_dict().items() if not n.startswith('action_gate.')}
    assert all(torch.equal(v.cpu(),source['r2p_and_projections'][n]) for n,v in frozen.items())
    assert all(torch.equal(v.cpu(),source['tasknet_net_g'][n]) for n,v in task.net_g.state_dict().items())
    return len(frozen)

@torch.no_grad()
def probe(model,task,cfg,raw,device,label):
    before=rng_state();model.eval();task.eval();rows=[]
    for i,rawbatch in enumerate(staged.loader(cfg,raw,'bias',16)):
        guard_gpu()
        for j,micro in enumerate(study.microbatches(rawbatch)):
            batch=study.base.unpack_batch(micro,device);study.base.set_step_seed(cfg['seed'],'fixed_train_monitor',i,batch['sample_id'])
            out=staged.forward(model,task,batch)
            rows.append(dict(sample_id=batch['sample_id'][0],loss=float(cost_loss(out.action['raw_cost'],action_costs(out.t1,out.rgb_increment,batch['target']),model.cost_scale)),t1_tensor_sha256=hashlib.sha256(out.t1.cpu().contiguous().numpy().tobytes()).hexdigest(),candidate_tensor_sha256=hashlib.sha256(out.ungated_t2.cpu().contiguous().numpy().tobytes()).hexdigest()))
            del out
    if label!='initial':
        ref=json.loads((OUT/'metrics/probe_initial.json').read_text())['per_image']
        assert [(r['sample_id'],r['t1_tensor_sha256'],r['candidate_tensor_sha256']) for r in rows]==[(r['sample_id'],r['t1_tensor_sha256'],r['candidate_tensor_sha256']) for r in ref]
    n=frozen_audit(model,task,cfg)
    json_save(OUT/'metrics'/f'probe_{label}.json',dict(training_role='bias',frames=len(rows),per_image=rows,mean_loss=float(np.mean([r['loss'] for r in rows])),all_non_judgement_tensors_equal_source=n,t1_and_candidate_hashes_fixed=True,used_for_selection=False,validation_used=False,test_used=False))
    restore_rng(before)
staged.probe=probe

def verify(cfg):
    seed(cfg['seed']);model,task,raw=components(cfg,torch.device('cpu'));legacy=source_gate(model,cfg,'cpu')
    source=torch.load(cfg['reuse']['checkpoint'],map_location='cpu',weights_only=False)
    for n,v in model.action_gate.state_dict().items():
        old=source['r2p_and_projections']['action_gate.'+n]
        if n=='body.0.weight':assert torch.equal(v[:,:43],old) and torch.count_nonzero(v[:,43:])==0
        else:assert torch.equal(v,old)
    n=frozen_audit(model,task,cfg)
    small=copy.deepcopy(cfg);small['data'].update(crop_size=32,workers=0)
    batch=study.base.unpack_batch(next(study.microbatches(next(iter(staged.loader(small,raw,'judgement',1))))),torch.device('cpu'))
    counts={'E':0,'D':0};hooks=[model.r2p.register_forward_hook(lambda *a:counts.__setitem__('E',counts['E']+1)),task.net_g.register_forward_hook(lambda *a:counts.__setitem__('D',counts['D']+1))]
    with torch.no_grad():seed(100);out=study.recurrent_forward(model,task,batch)
    assert counts=={'E':2,'D':2}
    first,candidate=out.t1.clone(),out.ungated_t2.clone()
    measured=action_costs(out.t1,out.rgb_increment,batch['target'])
    e=out.t1-batch['target'];d=out.rgb_increment
    formula=torch.cat([F.avg_pool2d((2*a*e*d+a*a*d.square()).mean(1,keepdim=True),4) for a in ACTIONS],1)
    assert torch.allclose(measured,formula,atol=1e-7,rtol=1e-3)
    staged.configure(model,task,'judgement');optimizer=staged.phase_optimizer(model,'judgement')
    state=staged.checkpoint(model,task,cfg,'judgement',0,optimizer);save(ROOT/'diagnostics/cpu_resume.pt',state)
    def update():
        optimizer.zero_grad(set_to_none=True);prediction=model.action_gate(*args(out,batch))
        cost_loss(prediction['raw_cost'],measured,model.cost_scale).backward()
        assert all(p.grad is None for name,p in model.named_parameters() if not name.startswith('action_gate.'))
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.action_gate.parameters())
        nn.utils.clip_grad_norm_(model.action_gate.parameters(),1);optimizer.step()
        return {k:v.clone() for k,v in model.action_gate.state_dict().items()},torch.rand(7)
    restore_rng(state['rng']);expected,rand1=update()
    restored=staged.restore(model,task,ROOT/'diagnostics/cpu_resume.pt',cfg);optimizer.load_state_dict(restored['optimizer']);restore_rng(restored['rng']);actual,rand2=update()
    assert all(torch.equal(v,actual[k]) for k,v in expected.items()) and torch.equal(rand1,rand2)
    with torch.no_grad():
        seed(100);after=staged.forward(model,task,batch)
        changed=dict(batch,target=torch.rand_like(batch['target']));seed(100);replacement=staged.forward(model,task,changed)
        assert torch.equal(first,after.t1) and torch.equal(candidate,after.ungated_t2)
        assert torch.equal(after.t2,replacement.t2) and torch.equal(after.ungated_t2,replacement.ungated_t2)
        masked=staged.gate(model,after,batch,'mask_zero');assert torch.equal(first,masked.t2)
    frozen_audit(model,task,cfg)
    for h in hooks:h.remove()
    json_save(ROOT/'diagnostics/gain_cpu_checks.json',dict(passed=True,source_tensors=len(source['r2p_and_projections']),frozen_tensors=n,all_net_g_equal_source=True,actual_rounds={'E':2,'D':2},inherited_judgement_weights_strict_except_7_zero_input_channels=True,gt_replacement_output_invariant=True,t1_and_candidate_fixed_after_optimizer_update=True,action_cost_algebra_exact=True,optimizer_rng_restore_exact=True,judgement_gradient_isolated=True,mask_zero_exact=True,diagnostic_updates_discarded=True))
    (ROOT/'diagnostics/cpu_resume.pt').unlink();del model,task,optimizer;gc.collect()

def benchmark(cfg,device):
    model,task,raw=components(cfg,device);staged.configure(model,task,'judgement');opt=staged.phase_optimizer(model,'judgement');dur=[]
    for i,rawbatch in enumerate(staged.loader(cfg,raw,'judgement',4)):
        guard_gpu();tick=time.perf_counter();opt.zero_grad(set_to_none=True)
        for micro in study.microbatches(rawbatch):
            batch=study.base.unpack_batch(micro,device);study.base.set_step_seed(cfg['seed'],'preflight_judge',i,batch['sample_id'])
            with torch.no_grad():out=study.recurrent_forward(model,task,batch)
            out=staged.gate(model,out,batch);(cost_loss(out.action['raw_cost'],action_costs(out.t1,out.rgb_increment,batch['target']),model.cost_scale)/2).backward();del out
        nn.utils.clip_grad_norm_(model.action_gate.parameters(),1);opt.step();torch.cuda.synchronize();dur.append(time.perf_counter()-tick)
    del opt,model,task;gc.collect();torch.cuda.empty_cache();model,task,raw=components(cfg,device)
    from rgb_final_confirmation.data import load_manifest,select_rows,PairedBranchDataset
    from PIL import Image
    rows=select_rows(load_manifest(cfg['paths']['manifest']),'validation');sizes=[]
    for i,row in enumerate(rows):
        with Image.open(row.official_rgb) as im:sizes.append((im.width*im.height,i))
    sample=PairedBranchDataset(rows,group='A1',crop_size=None)[max(sizes)[1]]
    batch={k:v[None].to(device) if torch.is_tensor(v) else [v] for k,v in sample.items()}
    guard_gpu();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    with torch.no_grad():out=staged.forward(model,task,batch)
    torch.cuda.synchronize();seconds=time.perf_counter()-tick
    result=dict(judgement_step_seconds=float(np.median(dur[1:])),full_image_seconds=seconds,full_image_shape=list(batch['polar'].shape),peak_vram_mb=torch.cuda.max_memory_allocated()/2**20,original_size=True,tiling=False,preflight_updates_discarded=True)
    json_save(ROOT/'diagnostics/gain_benchmark.json',result)

def matrix_features(model,legacy,out,batch):
    x=model.action_gate.inputs(*args(out,batch)).detach()
    assert x.shape[1]==50
    old=x.clone();a=out.t1.clamp(0,1);p=out.ungated_t2.clamp(0,1)
    old[:,3:12]=torch.cat([F.avg_pool2d(z,4) for z in (a,p,p-a)],1);old[:,43:]=0
    return x,old

@torch.no_grad()
def crop_records(model,task,legacy,cfg,raw,device,role):
    rows=[];oldrows=[];targets=[];scenes=[];audits=[];ids=[]
    for i,rawbatch in enumerate(staged.loader(cfg,raw,role,cfg['gain_study']['diagnostic_steps'][role])):
        for j,micro in enumerate(study.microbatches(rawbatch)):
            guard_gpu();batch=study.base.unpack_batch(micro,device);study.base.set_step_seed(cfg['seed'],'gain_audit_'+role,i,batch['sample_id']);out=study.recurrent_forward(model,task,batch)
            x,old=matrix_features(model,legacy,out,batch);y=action_costs(out.t1,out.rgb_increment,batch['target'])[:,1:]/model.cost_scale
            indices=np.random.default_rng(cfg['seed']+i*2+j).choice(x.shape[-2]*x.shape[-1],min(cfg['gain_study']['sampled_cells_per_crop'],x.shape[-2]*x.shape[-1]),replace=False)
            rows.append(x[0].permute(1,2,0).reshape(-1,50)[indices].cpu().numpy());oldrows.append(old[0].permute(1,2,0).reshape(-1,50)[indices].cpu().numpy());targets.append(y[0].permute(1,2,0).reshape(-1,3)[indices].cpu().numpy())
            scene=batch['scene'][0];scenes += [scene]*len(indices);ids.append(batch['sample_id'][0])
            oldpred=legacy(*args(out,batch))['raw_cost'];clipped=((out.t1<0)|(out.t1>1)|(out.ungated_t2<0)|(out.ungated_t2>1)).float().mean()
            audits.append(dict(sample_id=ids[-1],scene=scene,clipped_fraction=float(clipped),legacy_cost_mae=float((oldpred-y).abs().mean()),delta_energy=float(x[:,43:46].mean()),squared_mean_delta=float(x[:,9:12].square().mean())))
            del out,x,old,y
    payload=dict(x=np.concatenate(rows),legacy=np.concatenate(oldrows),y=np.concatenate(targets),scene=np.asarray(scenes))
    cache=ROOT/'diagnostics/feature_cache';cache.mkdir(exist_ok=True);np.savez_compressed(cache/f'{role}.npz',**payload)
    json_save(ROOT/'diagnostics'/f'{role}_input_audit.json',dict(role=role,crops=len(ids),sample_ids=ids,per_crop=audits,GT_only_in_labels=True,scene_role_fixed=True))
    return payload

def fit_ridge(data,columns,key='x'):
    x=data[key][:,columns].astype(np.float64);y=data['y'].astype(np.float64)
    mean=x.mean(0);sd=np.maximum(x.std(0),1e-5);z=(x-mean)/sd
    ym=y.mean(0);cov=z.T@z/len(z);weights=np.linalg.solve(cov+.01*np.eye(len(columns)),z.T@(y-ym)/len(z))
    return dict(columns=columns,mean=mean,sd=sd,weights=weights,intercept=ym,key=key)
def predict(probe,x):return (x[...,probe['columns']]-probe['mean'])/probe['sd']@probe['weights']+probe['intercept']

def risk_curve(predicted,actual,scenes,coverage,scale,weights=None):
    action=predicted.argmin(1);score=predicted.min(1);n=max(1,int(len(score)*coverage));order=np.argsort(score,kind='stable')[:n]
    weights=np.ones(len(actual)) if weights is None else weights
    chosen=actual[np.arange(len(actual)),action];gain=-float((chosen[order]*weights[order]).sum())/weights.sum()*scale
    fractions=np.bincount(action[order],weights=weights[order],minlength=3)/weights[order].sum()
    random_gain=-float((np.average(actual,axis=0,weights=weights)*fractions).sum())*weights[order].sum()/weights.sum()*scale
    byscene={}
    for scene in sorted(set(scenes)):
        subset=order[scenes[order]==scene];den=weights[scenes==scene].sum();byscene[str(scene)]=-float((chosen[subset]*weights[subset]).sum())/den*scale
    return dict(coverage=float(weights[order].sum()/weights.sum()),region_coverage=n/len(actual),mean_mse_improvement=gain,same_action_random_expected_mse_improvement=random_gain,harmful_accepted_rate=float(np.average(chosen[order]>.02,weights=weights[order])),by_scene_gain=byscene,diagnostic_forced_coverage_not_deployment=True)

@torch.no_grad()
def diagnostic(cfg,device):
    model,task,raw=components(cfg,device);legacy=source_gate(model,cfg,device)
    data={role:crop_records(model,task,legacy,cfg,raw,device,role) for role in ('judgement','bias','safety')}
    groups={'legacy':(list(range(43)),'legacy'),'rgb_float':(list(range(12))+list(range(43,50)),'x'),'cross_float':(list(range(12))+list(range(17,50)),'x'),'polar_float':(list(range(50)),'x')}
    probes={name:fit_ridge(data['judgement'],columns,key) for name,(columns,key) in groups.items()};reports={}
    for name,p in probes.items():
        bias=(data['bias']['y']-predict(p,data['bias'][p['key']])).mean(0);p['intercept']+=bias
        pred=predict(p,data['safety'][p['key']]);target=data['safety']['y'];curves=[risk_curve(pred,target,data['safety']['scene'],c,model.cost_scale) for c in cfg['gain_study']['coverage_grid']]
        reports[name]=dict(cost_mae=float(np.abs(pred-target).mean())*model.cost_scale,action_spearman=[correlation(-pred[:,j],-target[:,j]) for j in range(3)],risk_coverage=curves,feature_columns=p['columns'])
    check=next(r for r in reports['polar_float']['risk_coverage'] if abs(r['coverage']-.1)<1e-3)
    positive=sum(v>0 for v in check['by_scene_gain'].values());eligible=check['mean_mse_improvement']>0 and check['mean_mse_improvement']>check['same_action_random_expected_mse_improvement'] and positive>=3 and len(check['by_scene_gain'])==4
    receipt=dict(training_allowed=bool(eligible),selected_primary_change='float inputs and within-cell delta statistics' if eligible else None,safety_signal=check,positive_safety_scenes=positive,rule=cfg['gain_study']['train_gate_rule'],validation_used_for_trigger=False,test_used=False,probes=reports,diagnostic_failure_not_impossibility_proof=True)
    json_save(ROOT/'diagnostics/training_trigger.json',receipt)
    # Save fitted coefficients and exact sampled features for CPU re-verification.
    np.savez_compressed(ROOT/'diagnostics/ridge_probes.npz',**{name+'_'+key:value for name,p in probes.items() for key,value in p.items() if key!='key'})
    from rgb_final_confirmation.evaluation import psnr,ssim
    hashes={r['sample_id']:r['t1_tensor_sha256'] for r in json.loads((ROOT/'diagnostics/source_oracle_validation872.json').read_text())['per_image']}
    full=[];maps=ROOT/'diagnostics/validation_action_maps';maps.mkdir(exist_ok=True)
    for i,rawbatch in enumerate(study.base.split_loader(cfg,'validation')):
        guard_gpu();batch=study.base.unpack_batch(rawbatch,device);study.base.set_step_seed(cfg['seed'],'validation',i,batch['sample_id']);out=study.recurrent_forward(model,task,batch)
        digest=hashlib.sha256(out.t1.cpu().contiguous().numpy().tobytes()).hexdigest();assert digest==hashes[batch['sample_id'][0]]
        old=legacy(*args(out,batch),model.cost_bias,model.safety_margin)
        cost=action_costs(out.t1,out.rgb_increment,batch['target'])[:,1:].cpu().numpy()[0];scores=old['corrected_cost'].cpu().numpy()[0]*model.cost_scale
        x,xold=matrix_features(model,legacy,out,batch);x=x[0].permute(1,2,0).cpu().numpy();xold=xold[0].permute(1,2,0).cpu().numpy();target=cost.transpose(1,2,0).reshape(-1,3)/model.cost_scale
        cellscene=np.full(len(target),batch['scene'][0]);preds={name:predict(p,xold if p['key']=='legacy' else x).reshape(-1,3) for name,p in probes.items()}
        curves={name:[risk_curve(pred,target,cellscene,c,model.cost_scale) for c in cfg['gain_study']['coverage_grid']] for name,pred in preds.items()}
        curves['source_head']=[risk_curve(scores.transpose(1,2,0).reshape(-1,3)/model.cost_scale,target,cellscene,c,model.cost_scale) for c in cfg['gain_study']['coverage_grid']]
        np.savez_compressed(maps/f'{i:04d}.npz',actual_action_cost=cost,source_predicted_cost=scores,float_probe_predicted_cost=preds['polar_float'].reshape(*x.shape[:2],3)*model.cost_scale,sample_id=batch['sample_id'][0],scene=batch['scene'][0],shape=np.asarray(out.t1.shape),t1_tensor_sha256=digest,candidate_tensor_sha256=hashlib.sha256(out.ungated_t2.cpu().contiguous().numpy().tobytes()).hexdigest())
        # One predeclared10% output is only a diagnostic, never the deployed T2.
        prediction=preds['polar_float'];choice=prediction.argmin(1);order=np.argsort(prediction.min(1),kind='stable')[:max(1,len(choice)//10)];alpha=np.zeros(len(choice),np.float32);alpha[order]=np.asarray(ACTIONS[1:])[choice[order]]
        mask=F.interpolate(torch.from_numpy(alpha.reshape(1,1,*x.shape[:2])).to(device),size=out.t1.shape[-2:],mode='nearest');image=torch.where(mask==0,out.t1,out.t1+mask*out.rgb_increment)
        multiscale={};base_mse=float((out.t1-batch['target']).square().mean())
        for side in (16,32):
            factor=side//4
            def aggregate(value):
                t=torch.from_numpy(value.reshape(*x.shape[:2],3).transpose(2,0,1))[None]
                return F.avg_pool2d(t,factor,stride=factor,ceil_mode=True,count_include_pad=False)[0].permute(1,2,0).numpy()
            t=aggregate(target);w=F.avg_pool2d(torch.ones(1,1,*x.shape[:2]),factor,stride=factor,ceil_mode=True,divisor_override=1)[0,0].numpy().flatten()
            multiscale[str(side)]={}
            for name,p in preds.items():
                p=aggregate(p).reshape(-1,3);a=t.reshape(-1,3);scene=np.full(len(a),batch['scene'][0])
                records=[risk_curve(p,a,scene,c,model.cost_scale,w) for c in cfg['gain_study']['coverage_grid']]
                for record in records:record['stitched_psnr_from_exact_mse']=float(-10*np.log10(max(base_mse-record['mean_mse_improvement'],1e-12)))
                multiscale[str(side)][name]=records
        full.append(dict(sample_id=batch['sample_id'][0],scene=batch['scene'][0],t1_psnr=psnr(out.t1,batch['target']),t1_ssim=ssim(out.t1,batch['target']),probe10_psnr=psnr(image,batch['target']),probe10_ssim=ssim(image,batch['target']),curves=curves,multiscale_curves=multiscale))
        del out,x,xold,image,mask
        if (i+1)%20==0:json_save(ROOT/'evaluation_progress.json',dict(split='source_predictability_validation',completed=i+1,total=872));print(json.dumps(dict(diagnostic_frames=i+1)),flush=True)
    assert len(full)==872
    json_save(ROOT/'diagnostics/source_predictability_validation872.json',dict(frames=872,per_image=full,training_trigger_computed_before_validation=True,validation_not_used_for_trigger=True,test_used=False,t1_all_source_hashes_equal=True,original_size=True,tiling=False,GT_only_in_labels_and_metrics=True,probe10_is_forced_coverage_diagnostic_not_method_score=True,means={key:float(np.mean([r[key] for r in full])) for key in ('t1_psnr','t1_ssim','probe10_psnr','probe10_ssim')}))
    frozen_audit(model,task,cfg)

def initialize(cfg,device):
    model,task,raw=components(cfg,device);state=staged.checkpoint(model,task,cfg,'initial',0)
    save(OUT/'checkpoints/initial.pt',state);save(OUT/'checkpoints/candidate_final.pt',state)
    json_save(OUT/'config/candidate_selection.json',dict(step=0,planned_steps=0,source_candidate_steps=2000,candidate_frozen=True,sha256=sha(OUT/'checkpoints/candidate_final.pt')))
    probe(model,task,cfg,raw,device,'initial')
    json_save(OUT/'config/weight_migration.json',dict(source_sha256=cfg['reuse']['sha256'],source_protocol_id=cfg['reuse']['source_protocol_id'],all_non_judgement_tensors_and_net_g_strict=True,inherited_judgement_body_equal_except_expanded_input_weights=True,extra_input_channels=7,extra_weights_zero=True,optimizer_rng_inherited=False,candidate_and_q_frozen_entire_run=True,pretraining_and_official_weights_lineage_only=True))

class FullCostCollector(staged.ActionCollector):
    def __init__(self,*a,**k):super().__init__(*a,**k);self.raw_rows=[]
    def observe(self,out,batch,scale):
        super().observe(out,batch,scale)
        actual=action_costs(out.t1,out.rgb_increment,batch['target'])[:,1:];pred=out.action['corrected_cost']*scale
        values=[]
        for j in range(3):
            p=pred[:,j].cpu().numpy().flatten();a=actual[:,j].cpu().numpy().flatten();stride=max(len(p)//4096,1)
            values.append(dict(action=ACTIONS[j+1],predicted_mean_gain=-float(p.mean()),actual_mean_gain=-float(a.mean()),cost_mae=float(np.abs(p-a).mean()),gain_spearman=correlation(-p[::stride],-a[::stride])))
        self.raw_rows.append(dict(sample_id=batch['sample_id'][0],scene=batch['scene'][0],actions=values))
        if self.split=='validation':
            idx=len(self.rows)-1;source=ROOT/'diagnostics/validation_action_maps'/f'{idx:04d}.npz'
            with np.load(source) as saved:
                assert saved['sample_id'].item()==batch['sample_id'][0]
                assert saved['candidate_tensor_sha256'].item()==hashlib.sha256(out.ungated_t2.cpu().contiguous().numpy().tobytes()).hexdigest()
    def finish(self):
        result=super().finish();result['all_nonzero_action_predictions']=self.raw_rows;result['candidate_hashes_match_source_validation']=self.split=='validation';return result
staged.ActionCollector=FullCostCollector
