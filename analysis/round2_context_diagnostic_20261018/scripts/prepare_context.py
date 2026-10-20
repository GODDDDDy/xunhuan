"""Lock existing samples and sources before any new image inference."""
import csv, hashlib, json, time
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
P33=PROJECT/'confidence_causal_completion_20261017'
P32=PROJECT/'confidence_correction_causal_20261017'
P30=PROJECT/'c_feedback_polar_loop_20261015'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    path=ROOT/'config/locked.json'
    assert not path.exists(),'Do not overwrite an existing locked protocol'
    for d in ['config','metrics','reports','logs','failures','operations']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    cfg=json.loads((P33/'config/execution_config_2026.json').read_text())
    manifest=Path(cfg['paths']['manifest'])
    with manifest.open() as f:rows={r['sample_id']:r for r in csv.DictReader(f)}
    train=json.loads((P32/'config/diagnostic_crops.json').read_text())
    val=json.loads((P32/'config/validation64.json').read_text())['sample_ids']
    assert len(train)==16 and len(val)==64
    items=[]
    for split,entries in [('validation64',[dict(sample_id=s) for s in val]),('train16',train)]:
        for v in entries:
            r=rows[v['sample_id']]
            with Image.open(r['gt_rgb']) as im:w,h=im.size
            assert h%8==w%8==0 and min(h,w)>=128
            top=v['top'] if split=='train16' else (h-128)//16*8
            left=v['left'] if split=='train16' else (w-128)//16*8
            items.append(dict(sample_id=r['sample_id'],split=split,scene=r['scene'],top=top,left=left,height=h,width=w))
    paths=[P33/'weights/final.pt',P32/'weights/fit_last.pt',P30/'weights/common_initial21.pt',
           P33/'config/locked.json',P32/'config/locked.json',P33/'config/execution_config_2026.json',
           P33/'config/calibration.json',P32/'config/diagnostic_crops.json',P32/'config/validation64.json',manifest,
           P33/'scripts/completion_run.py',P32/'scripts/causal_run.py',P30/'scripts/c_loop_run.py',
           P30/'scripts/study.py',PROJECT/'polar_gain_judgement_short_20261012/source/rgb_final_confirmation/polarfree.py',
           PROJECT/'deps/DfP/archs/diffusion.py',PROJECT/'deps/DfP/archs/latent_encoder_arch.py',ROOT/'PROTOCOL_ZH.md']
    receipt=lambda p:dict(path=str(p),sha256=sha(p))
    obj=dict(experiment=34,created_unix=time.time(),items=items,main_model='F',background_model='FitP',
             training_steps_new=0,test_used=False,max_wall_seconds=7200,
             scoring={'center96':[16,16,96,96],'all128':[0,0,128,128]},
             mu=json.loads((P33/'config/calibration.json').read_text())['mu'],
             sources=[receipt(p) for p in paths],code_sources=[receipt(p) for p in sorted((ROOT/'scripts').glob('*.py'))],
             reference=receipt(P33/'metrics/validation872.json'),
             action_threshold=dict(psnr=.05,min_positive_scenes=6,mean_ssim_min=0,full_gain_max=0))
    obj['protocol_id']=hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(protocol_id=obj['protocol_id'],items=len(items),mu=obj['mu'],locked=str(path))))

if __name__=='__main__':main()
