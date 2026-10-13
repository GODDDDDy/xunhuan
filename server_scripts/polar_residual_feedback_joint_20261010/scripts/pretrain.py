"""Train the new enhancement module for same-resolution polarization correction."""
import argparse,csv,gc,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from enhancer import PolarEnhancer
from pidsr_data import PolarDataset
from polar_loss import RecoveryLoss,stokes,circular,masked_mean
from common import ROOT,seed,save,json_save,sha,rng_state,restore_rng,guard_gpu

OUT=ROOT/'pretraining'

def loader(cfg,split='train',start=0,steps=None):
    effective=cfg['pretrain']['effective_batch']
    data=PolarDataset(cfg['paths']['pidsr_data'],split,
                      samples=(steps or cfg['pretrain']['steps'])*effective-start*effective,
                      start=start*effective,seed=cfg['seed'])
    return DataLoader(data,batch_size=cfg['pretrain']['microbatch'] if split=='train' else 1,
                      shuffle=False,num_workers=2 if split=='train' else 0,pin_memory=False,
                      generator=torch.Generator().manual_seed(cfg['seed']+712),
                      **({'prefetch_factor':1} if split=='train' else {}))

def learning_rate(cfg,step):
    continuation=cfg['pretrain'].get('continuation')
    if continuation and step>continuation['start_step']:
        fraction=min((step-continuation['start_step'])/max(cfg['pretrain']['steps']-continuation['start_step'],1),1)
        return 1e-6+(continuation['start_lr']-1e-6)*.5*(1+math.cos(math.pi*fraction))
    if continuation:
        total=continuation['original_total_steps']
    else:total=cfg['pretrain']['steps']
    warm=min(500,total//10)
    if step<=warm:return 3e-4*step/max(warm,1)
    fraction=min((step-warm)/max(total-warm,1),1)
    return 1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*fraction))

@torch.no_grad()
def evaluate(model,cfg,split,device,step=0):
    import sys
    sys.path.insert(0,cfg['paths']['source_root'])
    from rgb_final_confirmation.evaluation import psnr,ssim
    model.eval();rows=[];criterion=RecoveryLoss().to(device)
    for batch in loader(cfg,split):
        if device.type=='cuda':guard_gpu()
        x=batch['polar'].to(device);target=batch['target'].to(device)
        # Same FP32 inference for validation/test; no tiles or hidden resizing.
        raw_prediction=model.pretrain_forward(x)
        recovery_loss=float(criterion(raw_prediction.float(),target.float()))
        prediction=raw_prediction.clamp(0,1)
        angles={}
        input_angles={}
        for i,a in enumerate((0,45,90,135)):
            sl=slice(i*3,i*3+3)
            angles[str(a)]={'psnr':psnr(prediction[:,sl],target[:,sl]),'ssim':ssim(prediction[:,sl],target[:,sl])}
            input_angles[str(a)]={'psnr':psnr(x[:,sl],target[:,sl]),'ssim':ssim(x[:,sl],target[:,sl])}
        s,g=stokes(prediction),stokes(target)
        original_s=stokes(x)
        row={'scene':batch['scene'][0],'angles':angles,'input_angles':input_angles,'recovery_loss':recovery_loss,
             'input_psnr':float(np.mean([v['psnr'] for v in input_angles.values()])),
             'input_ssim':float(np.mean([v['ssim'] for v in input_angles.values()])),
             'input_mae_DoLP_valid':float(masked_mean((original_s[3]-g[3]).abs(),g[5])),
             'input_mae_AoLP_degrees_valid':float(masked_mean(circular(original_s[4],g[4]),g[6])*180/math.pi),
             'psnr':float(np.mean([v['psnr'] for v in angles.values()])),
             'ssim':float(np.mean([v['ssim'] for v in angles.values()])),
             'mae_S0':float((s[0]-g[0]).abs().mean()),'mae_S1':float((s[1]-g[1]).abs().mean()),
             'mae_S2':float((s[2]-g[2]).abs().mean()),
             'mae_DoLP_valid':float(masked_mean((s[3]-g[3]).abs(),g[5])),
             'mae_AoLP_degrees_valid':float(masked_mean(circular(s[4],g[4]),g[6])*180/math.pi),
             'valid_DoLP_elements':int(g[5].sum()),'valid_AoLP_elements':int(g[6].sum())}
        rows.append(row)
    expected=10 if split=='validation' else 30
    assert len(rows)==len({r['scene'] for r in rows})==expected
    result={'split':split,'scenes':expected,'step':step,'psnr':float(np.mean([r['psnr'] for r in rows])),
            'recovery_loss':float(np.mean([r['recovery_loss'] for r in rows])),
            'ssim':float(np.mean([r['ssim'] for r in rows])), 'per_scene':rows,
            'input_psnr':float(np.mean([r['input_psnr'] for r in rows])),
            'input_ssim':float(np.mean([r['input_ssim'] for r in rows])),
            'grid':'same-resolution correction; official 1024x1024 HR ROI -> work256',
            'selection':'validation only; test never selects checkpoint'}
    json_save(OUT/'metrics'/f'{split}_step_{step}.json',result)
    return result

def convergence_check(cfg,step):
    """A preregistered validation stability heuristic, not proof of an optimum."""
    rule=cfg['pretrain'].get('convergence')
    if not rule:return {'stable':False,'enabled':False}
    history=[json.loads(p.read_text()) for p in (OUT/'metrics').glob('validation_step_*.json')]
    history=sorted((x for x in history if x['step']<=step and 'recovery_loss' in x),key=lambda x:x['step'])
    result={'enabled':True,'stable':False,'step':step,'minimum_steps':rule['minimum_steps'],
            'maximum_steps':cfg['pretrain']['steps'],'rule':rule,'test_used':False,
            'interpretation':'validation stability only; no guarantee of global or distributional convergence'}
    if step<rule['minimum_steps'] or len(history)<rule['window']:
        result['reason']='minimum updates or validation window not reached';return result
    recent=history[-rule['window']:]
    ps=np.array([x['psnr'] for x in recent]);ls=np.array([x['recovery_loss'] for x in recent])
    xs=np.array([x['step'] for x in recent])/1000
    span=float(np.ptp(ps));slope=float(np.polyfit(xs,ps,1)[0]);relative_loss_span=float(np.ptp(ls)/max(float(ls.mean()),1e-12))
    result.update(validation_steps=[x['step'] for x in recent],psnr_span=span,psnr_slope_db_per_1000=slope,
                  recovery_loss_relative_span=relative_loss_span,
                  stable=span<=rule['psnr_span_db'] and abs(slope)<=rule['psnr_slope_db_per_1000'] and relative_loss_span<=rule['recovery_loss_relative_span'])
    result['reason']='validation stabilized' if result['stable'] else 'validation still changing or oscillating'
    return result

def make_model(device):
    model=PolarEnhancer().to(device);model.activation_checkpoint=True
    for n,p in model.named_parameters():
        if n.startswith('adapters.') or ('.ffn.dwconv.' in n and not n.startswith('mid_refine.')):
            p.requires_grad_(False)
    return model

def benchmark(cfg,device):
    seed(cfg['seed']);model=make_model(device);criterion=RecoveryLoss().to(device)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=3e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    times=[];accum=8//cfg['pretrain']['microbatch'];model.train()
    it=iter(loader(cfg,steps=4))
    for step in range(4):
        guard_gpu();tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            batch=next(it);x=batch['polar'].to(device);t=batch['target'].to(device)
            with torch.autocast(device.type,enabled=device.type=='cuda',dtype=torch.bfloat16):p=model.pretrain_forward(x)
            with torch.autocast(device.type,enabled=False):loss=criterion(p,t)/accum
            if not torch.isfinite(loss):raise FloatingPointError('Pretrain benchmark nonfinite')
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
        scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
        times.append(time.perf_counter()-tick)
    # Evaluation timing uses a held-out validation scene, no test inference.
    b=next(iter(loader(cfg,'validation')));model.eval()
    with torch.no_grad():
        tick=time.perf_counter();model.pretrain_forward(b['polar'].to(device));torch.cuda.synchronize()
        eval_seconds=time.perf_counter()-tick
    result={'step_seconds':float(np.median(times[1:])),'validation_scene_seconds':eval_seconds,
            'peak_vram_mb':torch.cuda.max_memory_allocated()/2**20,'microbatch':cfg['pretrain']['microbatch']}
    json_save(ROOT/'diagnostics/pretrain_benchmark.json',result)
    del model,optimizer,scaler,it;gc.collect();torch.cuda.empty_cache();return result

def train(cfg,device):
    seed(cfg['seed']);model=make_model(device);criterion=RecoveryLoss().to(device)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=3e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    start=0;best=-float('inf');last=OUT/'checkpoints/last.pt'
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False)
        if ck['protocol_id']!=cfg['protocol_id']:
            lineage=cfg['pretrain']['continuation']
            assert ck['protocol_id']==lineage['parent_protocol_id'] and ck['step']==lineage['start_step']
            assert sha(last)==lineage['checkpoint_sha256']
        model.load_state_dict(ck['enhancer'],strict=True);optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler']);start=ck['step'];best=ck['best_validation'];restore_rng(ck['rng'])
    data=iter(loader(cfg,start=start));accum=8//cfg['pretrain']['microbatch']
    log=OUT/'logs/train.csv';log.parent.mkdir(parents=True,exist_ok=True)
    step=start
    while step<cfg['pretrain']['steps'] and time.time()<cfg['pretrain']['deadline_unix']:
        guard_gpu();step+=1;tick=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
        lr=learning_rate(cfg,step)
        for group in optimizer.param_groups:group['lr']=lr
        total=0.;scenes=[]
        for _ in range(accum):
            batch=next(data);scenes+=batch['scene'];x=batch['polar'].to(device);t=batch['target'].to(device)
            with torch.autocast(device.type,enabled=device.type=='cuda',dtype=torch.bfloat16):p=model.pretrain_forward(x)
            with torch.autocast(device.type,enabled=False):loss=criterion(p,t)/accum
            if not torch.isfinite(loss):raise FloatingPointError(f'Pretrain nonfinite step{step}')
            total+=float(loss.detach());scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True))
        scaler.step(optimizer);scaler.update()
        fields=dict(step=step,loss=total,grad_norm=norm,lr=lr,seconds=time.perf_counter()-tick,scenes=';'.join(scenes))
        exists=log.exists()
        with log.open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields.keys())
            if not exists:writer.writeheader()
            writer.writerow(fields)
        if step%50==0:print(json.dumps(fields),flush=True)
        stop=time.time()>=cfg['pretrain']['deadline_unix']
        if step%1000==0 or step==cfg['pretrain']['steps'] or stop:
            metrics=evaluate(model,cfg,'validation',device,step)
            improved=metrics['psnr']>best
            best=max(best,metrics['psnr'])
            state=dict(step=step,enhancer=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                       rng=rng_state(),best_validation=best,validation=metrics,protocol_id=cfg['protocol_id'],
                       pretraining_task='same-resolution polarization correction, no RGB/GT guide')
            save(last,state);save(OUT/'checkpoints'/f'step_{step}.pt',state)
            if improved:save(OUT/'checkpoints/selected.pt',state)
            decision=convergence_check(cfg,step)
            decision['time_limit_reached']=stop
            json_save(OUT/'convergence.json',decision)
            if stop or decision.get('stable'):break
    if not (OUT/'checkpoints/selected.pt').exists():raise RuntimeError('No evaluated pretraining checkpoint')
    chosen=torch.load(OUT/'checkpoints/selected.pt',map_location='cpu',weights_only=False)
    json_save(OUT/'selection.json',dict(step=chosen['step'],sha256=sha(OUT/'checkpoints/selected.pt'),
                                      validation_psnr=chosen['validation']['psnr'],selection='best validation mean angle PSNR',
                                      updates=step,test_read_for_selection=False,
                                      convergence=json.loads((OUT/'convergence.json').read_text()),
                                      parent_protocol_id=cfg['pretrain'].get('continuation',{}).get('parent_protocol_id'),
                                      selected_checkpoint_protocol_id=chosen['protocol_id']))
    model.load_state_dict(chosen['enhancer'],strict=True)
    result=evaluate(model,cfg,'test',device,chosen['step'])
    result.update(checkpoint_sha256=sha(OUT/'checkpoints/selected.pt'),test_completed=True)
    json_save(OUT/'metrics/final_test.json',result)
    return step
