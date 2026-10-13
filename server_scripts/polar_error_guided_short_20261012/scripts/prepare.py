"""Freeze source/data provenance before any GPU allocation."""
import csv,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT.parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def prepare():
    for n in ('config','diagnostics','logs','runtime/tmp','runtime/cache','pretraining','downstream/config','downstream/checkpoints'):
        (ROOT/n).mkdir(parents=True,exist_ok=True)
    prototype=json.loads((PARENT/'confidence_joint_20261009_v2/arms/A2_dual/run/config/run_config.json').read_text())
    paths=prototype['paths']
    for name in ('manifest','schedule','validation_proxy'):
        destination=ROOT/'config'/Path(paths[name]).name
        shutil.copy2(paths[name],destination);paths[name]=str(destination)
    paths['output_root']=str(ROOT/'downstream')
    paths['pidsr_data']=str(PARENT.parent/'research_data/PIDSR-dataset')
    paths['source_checkpoint']=prototype['study']['source_checkpoint']
    assert sha(paths['source_checkpoint'])=='eb17a5537bcd26583b73670a34a258f948056935f851b42c7a44eb62e1dd15ea'
    cfg=dict(schema_version=3,seed=2026,paths=paths,data=prototype['data'],
             source_sha256=prototype['study']['source_sha256'],
             scope='one new enhancer plus confidence K2; same-resolution PIDSR correction pretraining',
             pretrain=dict(steps=12000,effective_batch=8,microbatch=1,learning_rate=3e-4,minimum_steps=6000),
             downstream=dict(steps=8000,effective_batch=2,microbatch=1,minimum_steps=3000),
             confidence=dict(warmup_steps=500,calibration_training_crops=256),
             evaluation=dict(validation_frames=872,test_frames=188,original_size=True,tiles=False),
             selection='pretrain: best 10-scene validation mean angle PSNR; downstream: planned/time-only final')
    from pidsr_data import VAL_SCENES
    records=[]
    for p in sorted(Path(paths['pidsr_data']).rglob('RGB_*.png')):
        records.append(dict(path=str(p.relative_to(paths['pidsr_data'])),bytes=p.stat().st_size,sha256=sha(p)))
    assert len(records)==552
    dataset=dict(files=records,train_scenes=98,validation_scenes=VAL_SCENES,test_scenes=30,
                 channels='RGB',angle_order=[0,45,90,135],source_bitdepth=16,
                 task='CPFA/Bayer degradation -> same-resolution clean polarization; target never guide',
                 crops='HR512 -> work128 train; official HR1024 ROI -> work256 eval',
                 source_dataset='local F:/3080ti/PIDSR-dataset copied byte-for-byte')
    write(ROOT/'config/pidsr_data_manifest.json',dataset)
    with Path(paths['manifest']).open(newline='') as f:rows=list(csv.DictReader(f))
    counts={split:sum(r['split']==split for r in rows) for split in ('train','validation','test')}
    assert counts==dict(train=5440,validation=872,test=188),counts
    schedule=json.loads(Path(paths['schedule']).read_text())
    assert len(schedule['items'])>=16000
    sources=[Path(paths[k]) for k in ('base_config','manifest','schedule','validation_proxy','source_checkpoint')]
    sources+=list(Path(paths['source_root']).rglob('*.py'))
    lock=dict(code_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'scripts').rglob('*.py'))},
              sources_sha256={str(p):sha(p) for p in sources},
              pidsr_manifest_sha256=sha(ROOT/'config/pidsr_data_manifest.json'),downstream_split_counts=counts,
              official_checkpoint_unused=True,old_clean_RGB_guide_unused=True,
              references=['https://github.com/PRIS-CV/PIDSR','https://arxiv.org/html/2402.18134v2',
                          'https://github.com/fourson/Learning-to-Deblur-Polarized-Images',
                          'https://arxiv.org/html/2501.13134'])
    write(ROOT/'protocol_lock.json',lock);write(ROOT/'config/proposed_run.json',cfg)
    print(json.dumps(dict(prepared=True,pidsr_files=len(records),counts=counts,source_sha256=cfg['source_sha256'])))
if __name__=='__main__':prepare()
