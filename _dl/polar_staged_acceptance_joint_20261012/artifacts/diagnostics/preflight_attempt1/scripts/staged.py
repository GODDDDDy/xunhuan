"""One sequential candidate -> action judgement experiment; fixed first round."""
import csv,gc,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import ROOT,guard_gpu,json_save,save,sha,seed,rng_state,restore_rng,status
import study
from action_gate import ACTIONS,ActionCostGate,action_costs,compose,cost_loss
from quality_metrics import QualityCollector
from acceptance import local_error
OUT=ROOT/'downstream'

def components(cfg,device):
    model,task,_,raw=study.components(cfg,device,with_loss=False)
    model.action_gate=ActionCostGate(model.update_gate).to(device)
    model.cost_scale=1.;model.cost_bias=[0.,0.,0.];model.safety_margin=0.
    model.step=6000;model.eval();task.eval()
    model.requires_grad_(False);task.requires_grad_(False)
    return model,task,raw

def configure(model,task,phase):
    model.eval();task.eval();model.requires_grad_(False);task.requires_grad_(False)
    if phase=='candidate':model.feedback.requires_grad_(True).train()
    if phase=='judgement':model.action_gate.requires_grad_(True).train()

def loader(cfg,raw,phase,steps=None,start=0):
    local=json.loads(json.dumps(cfg));local['paths']['schedule']=cfg['staged']['schedules'][phase]
    return study.base.scheduled_loader(local,study.base.build_base_config(local),steps or cfg['staged']['steps'][phase],start)

def gate(model,out,batch,override=None):
    judgement=model.action_gate(batch['official_rgb'],study.polarization_summary(batch['polar']),out.t1,out.ungated_t2,out.error_scale,out.confidence_map,out.judgement_feature,out.judgement_state,model.cost_bias,model.safety_margin)
    mask=judgement['mask']
    if override=='mask_zero':mask=torch.zeros_like(mask)
    if override=='mask_one':mask=torch.ones_like(mask)
    out.t2=compose(out.t1,out.rgb_increment,mask);out.update_mask=mask;out.action=judgement
    return out

def forward(model,task,batch,diagnostic=None):
    out=study.recurrent_forward(model,task,batch,diagnostic=diagnostic)
    return gate(model,out,batch,diagnostic)

def checkpoint(model,task,cfg,phase,step,optimizer=None):
    state=dict(step=step,phase=phase,phase_step=step,r2p_and_projections=model.state_dict(),tasknet_net_g=task.net_g.state_dict(),tau=model.tau,cost_scale=model.cost_scale,cost_bias=model.cost_bias,safety_margin=model.safety_margin,protocol_id=cfg['protocol_id'],rng=rng_state(),selection='fixed phase endpoint or immutable time cutoff; no validation/test selection')
    if optimizer is not None:state['optimizer']=optimizer.state_dict()
    return state

def restore(model,task,path,cfg):
    state=torch.load(path,map_location='cpu',weights_only=False);assert state['protocol_id']==cfg['protocol_id']
    model.load_state_dict(state['r2p_and_projections'],strict=True);task.net_g.load_state_dict(state['tasknet_net_g'],strict=True)
    model.tau=state['tau'];model.cost_scale=state['cost_scale'];model.cost_bias=state['cost_bias'];model.safety_margin=state['safety_margin']
    return state

def candidate_loss(out,target):
    costs=action_costs(out.t1.detach(),out.rgb_increment,target)
    mse=(out.ungated_t2-target).square().mean()
    edge=0.
    for dim in (-1,-2):
        edge=edge+(out.ungated_t2.diff(dim=dim)-target.diff(dim=dim)).abs().mean()/2
    return mse+.02*edge+.1*costs[:,-1:].clamp_min(0).mean()

def verify_cpu(cfg):
    from candidate_diagnostic import cpu_check
    cpu_check();seed(cfg['seed'])
    model,task,raw=components(cfg,torch.device('cpu'))
    source=torch.load(cfg['reuse']['checkpoint'],map_location='cpu',weights_only=False)
    assert all(torch.equal(v,source['r2p_and_projections'][k]) for k,v in model.state_dict().items() if not k.startswith('action_gate.'))
    assert all(torch.equal(v,source['tasknet_net_g'][k]) for k,v in task.net_g.state_dict().items())
    small=dict(cfg);small['data']=dict(cfg['data'],crop_size=32,workers=0)
    batch=study.base.unpack_batch(next(study.microbatches(next(iter(study.base.scheduled_loader(small,study.base.build_base_config(small),1))))),torch.device('cpu'))
    configure(model,task,'candidate');counts={'E':0,'D':0}
    hooks=[model.r2p.register_forward_hook(lambda *args:counts.__setitem__('E',counts['E']+1)),task.net_g.register_forward_hook(lambda *args:counts.__setitem__('D',counts['D']+1))]
    seed(100);out=forward(model,task,batch);semantic=counts.copy()
    assert semantic=={'E':2,'D':2};assert torch.equal(out.t1,out.t2)
    first=out.t1.detach().clone();proposal=out.ungated_t2.detach().clone()
    candidate_loss(out,batch['target']).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.feedback.parameters())
    assert all(p.grad is None for p in model.action_gate.parameters())
    assert all(p.grad is None for p in model.r2p.parameters());assert all(p.grad is None for p in task.parameters())
    for p in model.parameters():p.grad=None
    configure(model,task,'judgement');seed(100)
    with torch.no_grad():out=study.recurrent_forward(model,task,batch)
    assert torch.equal(first,out.t1);assert torch.equal(proposal,out.ungated_t2)
    judgement=model.action_gate(batch['official_rgb'],study.polarization_summary(batch['polar']),out.t1,out.ungated_t2,out.error_scale,out.confidence_map,out.judgement_feature,out.judgement_state)
    cost_loss(judgement['raw_cost'],action_costs(out.t1,out.rgb_increment,batch['target']),.001).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.action_gate.parameters())
    assert all(p.grad is None for p in model.feedback.parameters())
    changed=dict(batch,target=torch.rand_like(batch['target']));seed(100)
    with torch.no_grad():replacement=forward(model,task,changed)
    assert torch.equal(first,replacement.t1);assert torch.equal(proposal,replacement.ungated_t2)
    assert torch.equal(replacement.t2,first)
    # Resume optimizer and RNG deterministically on the actual new judgement head.
    optimizer=torch.optim.Adam(model.action_gate.parameters(),lr=1e-4)
    state=checkpoint(model,task,cfg,'judgement',0,optimizer);save(ROOT/'diagnostics/cpu_resume.pt',state)
    restore_rng(state['rng']);expect=torch.rand(7);restore_rng(state['rng']);assert torch.equal(expect,torch.rand(7))
    reloaded=restore(model,task,ROOT/'diagnostics/cpu_resume.pt',cfg);optimizer.load_state_dict(reloaded['optimizer'])
    for hook in hooks:hook.remove()
    json_save(ROOT/'diagnostics/staged_cpu_checks.json',dict(passed=True,source_keys_equal=len(source['r2p_and_projections']),task_source_strict=True,first_round_frozen=True,actual_rounds=semantic,gt_replacement_no_output_change=True,initial_action_zero=True,candidate_and_judgement_gradients_separate=True,rng_restore_exact=True,optimizer_checkpoint_load=True))
    (ROOT/'diagnostics/cpu_resume.pt').unlink();del model,task;gc.collect()

def benchmark(cfg,device):
    model,task,raw=components(cfg,device);times={};peaks={}
    for phase in ('candidate','judgement'):
        configure(model,task,phase);module=model.feedback if phase=='candidate' else model.action_gate
        optimizer=torch.optim.Adam(module.parameters(),lr=1e-5 if phase=='candidate' else 1e-4)
        torch.cuda.reset_peak_memory_stats();durations=[]
        for i,rawbatch in enumerate(loader(cfg,raw,phase,3)):
            guard_gpu();tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            for micro in study.microbatches(rawbatch):
                batch=study.base.unpack_batch(micro,device);study.base.set_step_seed(cfg['seed'],'preflight_'+phase,i,batch['sample_id'])
                if phase=='candidate':out=forward(model,task,batch);loss=candidate_loss(out,batch['target'])
                else:
                    with torch.no_grad():out=study.recurrent_forward(model,task,batch)
                    out=gate(model,out,batch);loss=cost_loss(out.action['raw_cost'],action_costs(out.t1,out.rgb_increment,batch['target']),.001)
                (loss/2).backward();del out
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
            torch.nn.utils.clip_grad_norm_(module.parameters(),1,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize();durations.append(time.perf_counter()-tick)
        times[phase]=float(np.median(durations[1:]));peaks[phase]=torch.cuda.max_memory_allocated()/2**20
        del optimizer;gc.collect();torch.cuda.empty_cache()
    model.eval();model.requires_grad_(False)
    from rgb_final_confirmation.data import load_manifest,select_rows,PairedBranchDataset
    from PIL import Image
    rows=select_rows(load_manifest(cfg['paths']['manifest']),'validation');sizes=[]
    for i,row in enumerate(rows):
        with Image.open(row.official_rgb) as image:sizes.append((image.width*image.height,i))
    selected=max(sizes)[1];sample=PairedBranchDataset(rows,group='A1',crop_size=None)[selected]
    batch={k:v[None].to(device) if torch.is_tensor(v) else [v] for k,v in sample.items()};guard_gpu();tick=time.perf_counter()
    with torch.no_grad():out=forward(model,task,batch)
    torch.cuda.synchronize();seconds=time.perf_counter()-tick
    result=dict(step_seconds=times,train_peak_vram_mb=peaks,full_image_seconds=seconds,full_image_shape=list(batch['polar'].shape),eval_peak_vram_mb=torch.cuda.max_memory_allocated()/2**20,full_forward=True,tiling=False,preflight_updates_discarded=True)
    json_save(ROOT/'diagnostics/staged_benchmark.json',result);return result

@torch.no_grad()
def probe(model,task,cfg,raw,device,label):
    before=rng_state();model.eval();task.eval();rows=[]
    reference=json.loads((ROOT/'diagnostics/source_fixed_t1_reference.json').read_text());hashes={x['sample_id']:x['t1_tensor_sha256'] for x in reference['per_image']}
    from rgb_final_confirmation.evaluation import psnr,ssim
    for i,rawbatch in enumerate(study.base.proxy_loader(cfg,raw)):
        guard_gpu();batch=study.base.unpack_batch(rawbatch,device);study.base.set_step_seed(cfg['seed'],'fixed_system_probe',i,batch['sample_id'])
        out=forward(model,task,batch);digest=hashlib.sha256(out.t1.cpu().contiguous().numpy().tobytes()).hexdigest();assert digest==hashes[batch['sample_id'][0]]
        rows.append(dict(sample_id=batch['sample_id'][0],t1_tensor_sha256=digest,t1_psnr=psnr(out.t1,batch['target']),t2_psnr=psnr(out.t2,batch['target']),candidate_psnr=psnr(out.ungated_t2,batch['target']),t1_ssim=ssim(out.t1,batch['target']),t2_ssim=ssim(out.t2,batch['target'])))
        del out
    source=torch.load(cfg['reuse']['checkpoint'],map_location='cpu',weights_only=False)
    frozen={n:v for n,v in model.state_dict().items() if not n.startswith(('feedback.','action_gate.'))}
    assert all(torch.equal(v.cpu(),source['r2p_and_projections'][n]) for n,v in frozen.items());assert all(torch.equal(v.cpu(),source['tasknet_net_g'][n]) for n,v in task.net_g.state_dict().items())
    json_save(OUT/'metrics'/f'probe_{label}.json',dict(frames=len(rows),per_image=rows,first_round_hashes_equal_source=True,frozen_model_tensors_equal_source=True,net_g_tensors_equal_source=True,selection=False,test_used=False))
    restore_rng(before)

def train_phase(cfg,device,phase):
    model,task,raw=components(cfg,device);last=OUT/'checkpoints'/f'{phase}_last.pt'
    input_path=last if last.exists() else OUT/'checkpoints'/('initial.pt' if phase=='candidate' else 'candidate_final.pt')
    state=restore(model,task,input_path,cfg);start=state['phase_step'] if state['phase']==phase else 0
    configure(model,task,phase);module=model.feedback if phase=='candidate' else model.action_gate
    optimizer=torch.optim.Adam(module.parameters(),lr=1e-5 if phase=='candidate' else 1e-4)
    if state['phase']==phase and 'optimizer' in state:optimizer.load_state_dict(state['optimizer'])
    if phase=='judgement' and start==0:
        model.cost_scale=calibrate_scale(model,task,cfg,raw,device);seed(cfg['seed']+1)
    else:restore_rng(state['rng'])
    total=cfg['staged']['steps'][phase];phase_deadline=cfg['staged']['deadlines'][phase];step=start
    for rawbatch in loader(cfg,raw,phase,total,start):
        if time.time()>=phase_deadline:break
        guard_gpu();step+=1;tick=time.perf_counter();configure(model,task,phase);optimizer.zero_grad(set_to_none=True);loss_value=0.;ids=[]
        for micro in study.microbatches(rawbatch):
            batch=study.base.unpack_batch(micro,device);ids+=batch['sample_id'];study.base.set_step_seed(cfg['seed'],'training_'+phase,step,batch['sample_id'])
            if phase=='candidate':out=forward(model,task,batch);loss=candidate_loss(out,batch['target'])
            else:
                with torch.no_grad():out=study.recurrent_forward(model,task,batch)
                out=gate(model,out,batch);loss=cost_loss(out.action['raw_cost'],action_costs(out.t1,out.rgb_increment,batch['target']),model.cost_scale)
            assert torch.isfinite(loss);(loss/2).backward();loss_value+=float(loss.detach())/2;del out
        norm=float(torch.nn.utils.clip_grad_norm_(module.parameters(),1,error_if_nonfinite=True));optimizer.step()
        record=dict(phase=phase,step=step,loss=loss_value,grad_norm=norm,seconds=time.perf_counter()-tick,learning_rate=optimizer.param_groups[0]['lr'],core_lr=0.,task_lr=0.,sample_ids=';'.join(ids))
        p=OUT/'logs'/f'{phase}_train.csv';exists=p.exists()
        with p.open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=record)
            if not exists:writer.writeheader()
            writer.writerow(record)
        if step%50==0:print(json.dumps(record),flush=True)
        if step%500==0 or step==total:
            ck=checkpoint(model,task,cfg,phase,step,optimizer);save(last,ck);save(OUT/'checkpoints'/f'{phase}_step_{step}.pt',ck);status('training_'+phase,step=step,target_steps=total)
        if step%1000==0 or step==total:probe(model,task,cfg,raw,device,f'{phase}_{step}')
    if not last.exists():raise RuntimeError('No verifiable phase checkpoint')
    chosen=torch.load(last,map_location='cpu',weights_only=False);save(OUT/'checkpoints'/f'{phase}_final.pt',chosen)
    json_save(OUT/'config'/f'{phase}_selection.json',dict(step=chosen['phase_step'],planned_steps=total,sha256=sha(OUT/'checkpoints'/f'{phase}_final.pt'),rule='fixed endpoint or last predeclared time save',test_used=False))
    if chosen['phase_step']<cfg['staged']['minimum_steps'][phase]:raise RuntimeError('Phase failed to meet predeclared minimum; not a complete formal experiment')
    del model,task,optimizer;gc.collect();torch.cuda.empty_cache()

@torch.no_grad()
def calibrate_scale(model,task,cfg,raw,device):
    values=[];ids=[];before=rng_state()
    for i,rawbatch in enumerate(loader(cfg,raw,'judgement',128)):
        guard_gpu()
        for micro in study.microbatches(rawbatch):
            batch=study.base.unpack_batch(micro,device);study.base.set_step_seed(cfg['seed'],'cost_scale',i,batch['sample_id']);out=study.recurrent_forward(model,task,batch)
            values.append(action_costs(out.t1,out.rgb_increment,batch['target'])[:,1:].abs().flatten().cpu());ids+=batch['sample_id'];del out
    scale=max(float(torch.quantile(torch.cat(values),.95)),1e-5);json_save(OUT/'config/cost_scale.json',dict(scale=scale,rule='95th percentile absolute action MSE differences; floor1e-5',training_crops=len(ids),sample_ids=ids,test_used=False,validation_used=False))
    restore_rng(before);return scale

@torch.no_grad()
def calibration_records(model,task,cfg,raw,device,role):
    predicted=[];actual=[];ids=[]
    for i,rawbatch in enumerate(loader(cfg,raw,role)):
        guard_gpu()
        for micro in study.microbatches(rawbatch):
            batch=study.base.unpack_batch(micro,device);study.base.set_step_seed(cfg['seed'],'calibration_'+role,i,batch['sample_id']);out=study.recurrent_forward(model,task,batch);out=gate(model,out,batch)
            predicted.append(out.action['raw_cost'].permute(0,2,3,1).reshape(-1,3).cpu());actual.append((action_costs(out.t1,out.rgb_increment,batch['target'])[:,1:]/model.cost_scale).permute(0,2,3,1).reshape(-1,3).cpu());ids+=batch['sample_id'];del out
    return torch.cat(predicted),torch.cat(actual),ids

def calibrate(cfg,device):
    model,task,raw=components(cfg,device);state=restore(model,task,OUT/'checkpoints/judgement_final.pt',cfg);model.eval()
    predicted,actual,ids=calibration_records(model,task,cfg,raw,device,'bias');bias=(actual-predicted).mean(0);model.cost_bias=bias.tolist()
    predicted,actual,safety_ids=calibration_records(model,task,cfg,raw,device,'safety');predicted+=bias
    grid=[]
    for margin in cfg['staged']['safety_margin_grid']:
        cost=torch.cat((torch.zeros_like(predicted[:,:1]),predicted+margin),1);decision=cost.argmin(1);accepted=decision>0
        true=torch.cat((torch.zeros_like(actual[:,:1]),actual),1).gather(1,decision[:,None])[:,0]
        coverage=float(accepted.float().mean());harm=float((true[accepted]>.02).float().mean()) if accepted.any() else 0.
        gain=-float(true.mean())*model.cost_scale
        grid.append(dict(margin=margin,coverage=coverage,accepted_harmful_rate=harm,mean_mse_improvement=gain,eligible=gain>=0 and harm<=.2))
    eligible=[r for r in grid if r['eligible']]
    best=max(eligible,key=lambda r:(r['coverage'],-r['margin'])) if eligible else dict(margin=1e9,coverage=0.,reason='no eligible threshold; reject all')
    model.safety_margin=best['margin'];final=checkpoint(model,task,cfg,'calibrated_final',state['step']);final['candidate_steps']=json.loads((OUT/'config/candidate_selection.json').read_text())['step'];final['judgement_steps']=state['step'];save(OUT/'checkpoints/final.pt',final)
    json_save(OUT/'config/action_calibration.json',dict(cost_scale=model.cost_scale,bias=model.cost_bias,bias_sample_ids=ids,safety_sample_ids=safety_ids,grid=grid,chosen=best,validation_used=False,test_used=False,source_scenes_seen_by_inherited_weights=True))
    json_save(OUT/'selection.json',dict(sha256=sha(OUT/'checkpoints/final.pt'),candidate_steps=final['candidate_steps'],judgement_steps=final['judgement_steps'],protocol_id=cfg['protocol_id'],rule='fixed phase endpoints then predeclared training-only calibration',test_used=False))
    probe(model,task,cfg,raw,device,'calibrated_final')

class ActionCollector:
    def __init__(self,root,split,threshold):self.root=root;self.split=split;self.threshold=threshold;self.rows=[];self.saved=set();self.bins=[]
    def observe(self,out,batch,scale):
        actual=action_costs(out.t1,out.rgb_increment,batch['target']);cost=out.action['corrected_cost']*scale
        decision=out.action['decision'];selected=actual.gather(1,decision);accepted=decision>0;oracle=actual.argmin(1,keepdim=True);useful=actual.amin(1,keepdim=True)<-.02*scale
        good=local_error(out.t1,batch['target'])<=self.threshold
        final_cost=F.avg_pool2d((out.t2-batch['target']).square().mean(1,keepdim=True),4);first_cost=F.avg_pool2d((out.t1-batch['target']).square().mean(1,keepdim=True),4)
        keep=(out.update_mask==0).expand_as(out.t1);assert torch.equal(out.t1[keep],out.t2[keep])
        row=dict(sample_id=batch['sample_id'][0],scene=batch['scene'][0],cells=decision.numel(),good_cells=int(good.sum()),difficult_cells=int((~good).sum()),good_harmed=int((good&(final_cost-first_cost>.02*scale)).sum()),difficult_improved=int((~good&(first_cost-final_cost>.02*scale)).sum()),accepted=int(accepted.sum()),accepted_harmful=int((accepted&(selected>.02*scale)).sum()),useful=int(useful.sum()),rejected_useful=int((~accepted&useful).sum()),action_fractions=[float((decision==j).float().mean()) for j in range(4)],mask_mean=float(out.update_mask.mean()),protected_pixel_fraction=float((out.update_mask==0).float().mean()),cost_prediction_mae=float((cost-actual[:,1:]).abs().mean()),oracle_action_accuracy=float((decision==oracle).float().mean()))
        self.rows.append(row)
        selected_prediction=torch.cat((torch.zeros_like(cost[:,:1]),cost),1).gather(1,decision)
        flat_pred=(-selected_prediction).cpu().flatten().numpy();flat_gain=(-selected*scale/scale).cpu().flatten().numpy()
        self.bins.append((flat_pred,flat_gain))
        scene=batch['scene'][0]
        if scene not in self.saved:
            from PIL import Image
            root=self.root/'examples'/self.split;root.mkdir(parents=True,exist_ok=True)
            for name,value in [('T1',out.t1),('T2',out.t2),('candidate',out.ungated_t2),('GT',batch['target'])]:
                array=(value[0].detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8');Image.fromarray(array).save(root/f'{scene}_{name}.png')
            Image.fromarray((out.update_mask[0,0].cpu().numpy()*255).round().astype('uint8')).save(root/f'{scene}_action.png');self.saved.add(scene)
    def finish(self):
        totals={k:sum(r[k] for r in self.rows) for k in ['cells','good_cells','difficult_cells','good_harmed','difficult_improved','accepted','accepted_harmful','useful','rejected_useful']}
        predicted=np.concatenate([x[0] for x in self.bins]);gain=np.concatenate([x[1] for x in self.bins]);quantiles=np.quantile(predicted,np.linspace(0,1,11));bins=[]
        for i in range(10):
            mask=(predicted>=quantiles[i])&(predicted<=quantiles[i+1] if i==9 else predicted<quantiles[i+1]);bins.append(dict(lower=float(quantiles[i]),upper=float(quantiles[i+1]),cells=int(mask.sum()),predicted_gain=float(predicted[mask].mean()) if mask.any() else None,actual_gain=float(gain[mask].mean()) if mask.any() else None))
        return dict(per_image=self.rows,totals=totals,accepted_harmful_rate=totals['accepted_harmful']/max(totals['accepted'],1),rejected_useful_rate=totals['rejected_useful']/max(totals['useful'],1),good_region_harm_rate=totals['good_harmed']/max(totals['good_cells'],1),difficult_region_improvement_rate=totals['difficult_improved']/max(totals['difficult_cells'],1),protected_pixel_fraction=float(np.mean([r['protected_pixel_fraction'] for r in self.rows])),mean_mask=float(np.mean([r['mask_mean'] for r in self.rows])),calibration_bins=bins,definition='raw RGB MSE change > .02 training cost_scale; good region uses inherited train-only local-error threshold')

@torch.no_grad()
def evaluate(cfg,device,split):
    model,task,raw=components(cfg,device);state=restore(model,task,OUT/'checkpoints/final.pt',cfg);model.eval();task.eval()
    from rgb_final_confirmation.evaluation import psnr,ssim
    expected=872 if split=='validation' else 188;quality=QualityCollector(OUT,expected,model.tau);actions=ActionCollector(OUT,split,model.good_threshold);rows=[]
    for i,rawbatch in enumerate(study.base.split_loader(cfg,split)):
        if device.type=='cuda':guard_gpu()
        batch=study.base.unpack_batch(rawbatch,device);study.base.set_step_seed(cfg['seed'],'validation',i,batch['sample_id']);out=forward(model,task,batch)
        row=dict(sample_id=batch['sample_id'][0],scene=batch['scene'][0]);row.update({k:dict(psnr=psnr(v,batch['target']),ssim=ssim(v,batch['target'])) for k,v in [('round1',out.t1),('round2',out.t2),('ungated_candidate',out.ungated_t2)]});row['iteration_gain']={m:row['round2'][m]-row['round1'][m] for m in ('psnr','ssim')};rows.append(row)
        quality.observe(out,batch);actions.observe(out,batch,model.cost_scale);del out
        if (i+1)%20==0:json_save(ROOT/'evaluation_progress.json',dict(split=split,completed=i+1,total=expected,time_unix=time.time()));print(json.dumps(dict(split=split,frames=i+1)),flush=True)
    assert len(rows)==len({x['sample_id'] for x in rows})==expected
    keys=['round1','round2','ungated_candidate','iteration_gain'];scenes=[]
    for scene in sorted({x['scene'] for x in rows}):
        members=[x for x in rows if x['scene']==scene];scenes.append(dict(scene=scene,frames=len(members),**{k:{m:float(np.mean([x[k][m] for x in members])) for m in ('psnr','ssim')} for k in keys}))
    result=dict(split=split,frames=expected,checkpoint_sha256=sha(OUT/'checkpoints/final.pt'),candidate_steps=state['candidate_steps'],judgement_steps=state['judgement_steps'],protocol_id=cfg['protocol_id'],per_image=rows,per_scene=scenes,confidence=quality.finish(),action_acceptance=actions.finish(),original_size=True,tiling=False,device=str(device),test_used_for_selection=False)
    for k in keys:result[k]={**{f'image_mean_{m}':float(np.mean([x[k][m] for x in rows])) for m in ('psnr','ssim')},**{f'scene_macro_{m}':float(np.mean([x[k][m] for x in scenes])) for m in ('psnr','ssim')}}
    json_save(OUT/'metrics'/f'final_{split}.json',result)

@torch.no_grad()
def mechanism(cfg,device):
    model,task,raw=components(cfg,device);restore(model,task,OUT/'checkpoints/final.pt',cfg);model.eval()
    from rgb_final_confirmation.evaluation import psnr,ssim
    conditions=(None,'no_feedback','no_polar_correction','confidence_zero','confidence_one','mask_zero','mask_one');rows=[]
    for i,rawbatch in enumerate(study.base.proxy_loader(cfg,raw)):
        batch=study.base.unpack_batch(rawbatch,device);record=dict(sample_id=batch['sample_id'][0],conditions={})
        for condition in conditions:
            guard_gpu();study.base.set_step_seed(cfg['seed'],'mechanism_probe',i,batch['sample_id']);out=forward(model,task,batch,condition);record['conditions'][condition or 'normal']=dict(psnr=psnr(out.t2,batch['target']),ssim=ssim(out.t2,batch['target']),mask_mean=float(out.update_mask.mean()));del out
        rows.append(record);print(json.dumps(dict(mechanism_frame=i+1)),flush=True)
    json_save(OUT/'metrics/mechanism_validation.json',dict(frames=28,per_image=rows,test_used=False,matched_training_ablation=False,tuned_from_results=False,means={c or 'normal':{m:float(np.mean([x['conditions'][c or 'normal'][m] for x in rows])) for m in ('psnr','ssim','mask_mean')} for c in conditions}))

def initialize(cfg,device):
    model,task,raw=components(cfg,device);seed(cfg['seed']);save(OUT/'checkpoints/initial.pt',checkpoint(model,task,cfg,'initial',0));probe(model,task,cfg,raw,device,'initial')
    json_save(OUT/'config/weight_migration.json',dict(source_sha256=cfg['reuse']['sha256'],source_step=6000,source_protocol_id=cfg['reuse']['source_protocol_id'],all_inherited_model_and_task_tensors_equal=True,new_cost_output_zero=True,reused_judgement_feature_trunk=True,optimizer_and_rng_inherited=False,pretraining_and_official_weights_lineage_only=True))
