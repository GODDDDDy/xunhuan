"""One fixed-scale CPU pass over archived B; no new predictions or fitting."""
import csv,hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'c_feedback_polar_loop_20261015'
SOURCE=OLD.parent/'polar_gain_judgement_short_20261012/source'
sys.path.insert(0,str(SOURCE))
sys.path.insert(0,str(OLD/'scripts'))
from rgb_final_confirmation.data import _read_rgb
from quality_metrics import spearman_numpy

def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)

def pool(x):
    if x.ndim==3:x=x.mean(0)
    h,w=x.shape;assert h%4==w%4==0
    return x.reshape(h//4,4,w//4,4).mean((1,3))

def corr(a,b):
    value=spearman_numpy(a.ravel(),b.ravel())
    return float(value) if math.isfinite(value) else None

def dist(x):
    x=np.asarray(x,dtype=np.float64).ravel()
    if not len(x):return dict(mean=None,q10=None,q50=None,q90=None)
    q=np.quantile(x,[.1,.5,.9]);return dict(mean=float(x.mean()),q10=float(q[0]),q50=float(q[1]),q90=float(q[2]))

def flat(d,prefix=''):
    out={}
    for k,v in d.items():
        key=prefix+k
        if isinstance(v,dict):out.update(flat(v,key+'.'))
        elif isinstance(v,(int,float)) or v is None:out[key]=v
    return out

def aggregate(rows):
    data=[flat(r) for r in rows];keys=sorted(set().union(*(set(x) for x in data)))
    out={}
    for key in keys:
        vals=[x[key] for x in data if key in x and x[key] is not None]
        out[key]=dict(mean=float(np.mean(vals)) if vals else None,valid_images=len(vals))
    return out

def summarize(rows):
    scenes=sorted({r['scene'] for r in rows})
    by={s:aggregate([r for r in rows if r['scene']==s]) for s in scenes}
    macro={}
    for key in by[scenes[0]]:
        vals=[by[s][key]['mean'] for s in scenes if by[s][key]['mean'] is not None]
        macro[key]=dict(mean=float(np.mean(vals)) if vals else None,valid_scenes=len(vals),
                        scenes_positive=sum(v>0 for v in vals),scenes_negative=sum(v<0 for v in vals))
    return dict(frames=len(rows),scene_counts={s:sum(r['scene']==s for r in rows) for s in scenes},
                scene_equal=macro,image_equal=aggregate(rows),scenes=by,
                uncertainty='descriptive; cells are not independent replicates; one training seed')

def plot_sample(sid,t1,gt,c,e1,g,change):
    from PIL import Image,ImageDraw,ImageOps
    # Pillow previews only: all numerical analysis retains original float arrays.
    canvas=Image.new('RGB',(1290,820),'white');draw=ImageDraw.Draw(canvas)
    draw.text((12,8),sid+' | fixed first ID per scene; 4x4 grid',fill='black')
    def gray(z,maximum):
        u=np.clip(z/max(maximum,1e-30),0,1)
        return np.repeat((255*u)[...,None],3,axis=-1).astype('uint8')
    limit=max(float(np.quantile(abs(g),.99)),1e-30);v=np.clip(g/limit,-1,1)
    gain_rgb=np.stack((255*(1-np.maximum(-v,0)),255*(1-abs(v)),255*(1-np.maximum(v,0))),axis=-1).astype('uint8')
    panels=[(np.uint8(np.clip(t1,0,1).transpose(1,2,0)*255),'Fixed T1'),(np.uint8(gt.transpose(1,2,0)*255),'Ground truth'),
            (gray(c,1),'Predicted C: black=0, white=1'),
            (gray(e1,float(np.quantile(e1,.99))),'T1 MSE: white >= image 99th percentile'),
            (gain_rgb,f'MSE gain: red improves, blue worsens; +/-{limit:.3g}'),
            (gray(change,float(np.quantile(change,.99))),'RGB update: white >= image 99th percentile')]
    for i,(pixels,title) in enumerate(panels):
        x=10+(i%3)*430;y=40+(i//3)*385
        draw.text((x,y),title,fill='black')
        im=ImageOps.contain(Image.fromarray(pixels),(420,350))
        canvas.paste(im,(x,y+24))
    draw.text((10,805),'Resized previews; statistics use all native cells. Saturated preview scales do not clip statistics.',fill='black')
    target=ROOT/'reports/figures'/(sid.replace('/','_')+'.png');target.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(target)

def main():
    p=json.loads((ROOT/'config/plan.json').read_text())
    assert p['offline']['block']==4 and p['offline']['split']=='validation'
    source=json.loads((OLD/'metrics/validation872.json').read_text())
    reference={r['sample_id']:r for r in source['rows']}
    manifest={r['sample_id']:r for r in csv.DictReader((OLD/'config/manifest.csv').open())}
    rows=[];partial=ROOT/'metrics/offline_partial.json'
    if partial.exists():rows=json.loads(partial.read_text())['rows']
    done={r['sample_id'] for r in rows};assert len(done)==len(rows) and done<=set(reference)
    chosen=set(p['offline']['fixed_examples']);start=time.time()
    for sid in sorted(reference):
        if sid in done:continue
        with np.load(OLD/'examples'/(sid.replace('/','_')+'.npz')) as z:
            t1=z['t1'][0].astype(np.float64)
            t2=z['t2_B'][0].astype(np.float64)
            c=z['c'][0,0].astype(np.float64)
        gt=_read_rgb(manifest[sid]['gt_rgb']).numpy().astype(np.float64)
        assert t1.shape==t2.shape==gt.shape and c.shape==(t1.shape[-2]//4,t1.shape[-1]//4)
        assert all(np.isfinite(x).all() for x in (t1,t2,c,gt))
        a=t1-gt;b=t2-gt;d=t2-t1
        mse1=pool(a*a);mse2=pool(b*b);gain=mse1-mse2
        mae1=pool(abs(np.clip(t1,0,1)-gt));mae2=pool(abs(np.clip(t2,0,1)-gt));change=pool(abs(d))
        coupling=-pool(2*a*d+d*d);assert np.allclose(gain,coupling,atol=1e-15,rtol=1e-10)
        m1,m2=float(mse1.mean()),float(mse2.mean())
        ceiling=float(np.minimum(mse1,mse2).mean())
        psnr=lambda x:-10*math.log10(max(x,1e-30))
        psnr1,psnr2,psnrs=psnr(m1),psnr(m2),psnr(ceiling)
        assert abs(psnr1-reference[sid]['metrics']['T1']['psnr'])<=1e-4
        assert abs(psnr2-reference[sid]['metrics']['B']['psnr'])<=1e-4
        assert psnrs>=psnr1-1e-10 and psnrs>=psnr2-1e-10
        ordered=np.argsort(c.ravel(),kind='stable');groups=np.array_split(ordered,5)
        bins={}
        for i,ix in enumerate(groups):
            bins[str(i+1)]=dict(mean_C=float(c.ravel()[ix].mean()),mse_gain=float(gain.ravel()[ix].mean()),
                  mae_gain=float((mae1-mae2).ravel()[ix].mean()),positive_fraction=float((gain.ravel()[ix]>0).mean()),
                  mean_rgb_change=float(change.ravel()[ix].mean()),mean_first_mae=float(mae1.ravel()[ix].mean()))
        cp=np.exp(-np.maximum(mae1,.001)/p['oracle']['tau'])
        pos=gain>0;neg=gain<0
        row=dict(sample_id=sid,scene=reference[sid]['scene'],cells=int(c.size),
                 spearman=dict(C_first_MAE=corr(c,mae1),C_first_MSE=corr(c,mse1),
                               first_MSE_gain=corr(mse1,gain),C_mse_gain=corr(c,gain),C_mae_gain=corr(c,mae1-mae2)),
                 candidate=dict(actual_psnr_gain=psnr2-psnr1,selection_psnr_gain=psnrs-psnr1,
                                positive_fraction=float(pos.mean()),negative_fraction=float(neg.mean()),zero_fraction=float((gain==0).mean()),
                                mse_gain=float(gain.mean()),positive_mse_mass=float(np.maximum(gain,0).mean()),negative_mse_mass=float(np.maximum(-gain,0).mean()),
                                rgb_change_mean=float(change.mean()),rgb_change_below_1e_6=float((change<=1e-6).mean()),
                                mean_polar_change=reference[sid]['local']['B']['mean_polar_change']),
                 quintiles=bins,C_positive=dist(c[pos]),C_negative=dist(c[neg]),
                 change_positive=dist(change[pos]),change_negative=dist(change[neg]),
                 distributions=dict(pred_C=dist(c),oracle_C=dist(cp)),
                 C_tie_fraction=1-float(np.unique(c).size/c.size),
                 first_MAE=float(mae1.mean()),predicted_b_MAE=float(np.mean(abs(-p['oracle']['tau']*np.log(np.maximum(c,1e-300))-mae1))))
        rows.append(row);write(partial,dict(rows=rows,block=4,split='validation'))
        if sid in chosen:plot_sample(sid,t1,gt,c,mse1,gain,change)
        if len(rows)%16==0:
            write(ROOT/'offline_status.json',dict(stage='running',completed=len(rows),total=872,time_unix=time.time()))
            print(json.dumps(dict(completed=len(rows),total=872)),flush=True)
    assert len(rows)==872 and {r['sample_id'] for r in rows}==set(reference)
    summary=summarize(rows)
    write(ROOT/'metrics/offline_validation872.json',dict(rows=rows,summary=summary,protocol_id=p['protocol_id']))
    s=summary['scene_equal'];meaningful=s['candidate.selection_psnr_gain']['mean']>=.05
    result=dict(summary=summary,selection_resource_line=.05,fixed_candidate_selection_exceeds_resource_line=meaningful,
                conditional_training_gate=False,oracle_training_authorized_regardless_of_offline_result=True,
                fixed_examples=sorted(chosen),elapsed_seconds=time.time()-start,block=4,only_archived_predictions=True,
                inference_GT_use=False,selection_is_deployable=False)
    write(ROOT/'reports/offline_summary.json',result)
    # Human-readable tables remain on the server alongside machine-verifiable rows.
    text=['# Fixed-B offline diagnostic (4x4, validation872)','',
          '|Scene|Images|C vs first MAE|C vs MSE gain|Actual PSNR gain|GT block-selection PSNR gain|',
          '|---|---:|---:|---:|---:|---:|']
    for scene,v in summary['scenes'].items():
        fmt=lambda k:'NA' if v[k]['mean'] is None else f"{v[k]['mean']:.8f}"
        text.append(f"|{scene}|{summary['scene_counts'][scene]}|{fmt('spearman.C_first_MAE')}|{fmt('spearman.C_mse_gain')}|{fmt('candidate.actual_psnr_gain')}|{fmt('candidate.selection_psnr_gain')}|")
    text+=['','Selection is restricted to two fixed outputs per 4x4 block. It is not a bound on retraining or on SSIM.',
           'All correlations are descriptive, per-image then scene-equal; pixels are not independent replicates.',
           'Oracle training proceeds by explicit user authorization, regardless of this diagnostic.']
    (ROOT/'reports/offline_tables.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    write(ROOT/'offline_status.json',dict(stage='completed',frames=872,time_unix=time.time()))
    print(json.dumps(dict(completed=True,frames=872,selection_psnr_gain=s['candidate.selection_psnr_gain'],
                         C_first_MAE=s['spearman.C_first_MAE'],C_mse_gain=s['spearman.C_mse_gain'])),flush=True)

if __name__=='__main__':
    import torch
    torch.set_num_threads(2)
    main()
