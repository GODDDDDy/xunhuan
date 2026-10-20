"""Post-hoc CPU-only implementation review; preserve original experiment receipts."""
from pathlib import Path
import os,sys,json,hashlib,time,torch
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929/g6_enhancement_feedback_20261019')
OUT=ROOT/'reports/codex_review_20261019'
sys.path.insert(0,str(ROOT/'scripts'))
import audit_g6_feedback as audit
import run_g6_feedback as run

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return audit.sha(p)
def write_new(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def receipt(p):return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))

def state_checks():
    torch.set_num_threads(2);torch.manual_seed(381019)
    ck=torch.load(ROOT/'weights/final.pt',map_location='cpu',weights_only=False)
    image=torch.rand(1,3,64,64);t1=torch.rand_like(image)
    polar={'P2':torch.randn(1,96,32,32),'P3':torch.randn(1,192,16,16)}
    features={'T2':torch.randn(1,96,32,32),'T3':torch.randn(1,192,16,16)}
    result={}
    with torch.no_grad():
        for arm in ['B','R']:
            module=run.TaskFeedback().eval()
            module.load_state_dict({k[len('feedback.'):]:v for k,v in ck['blocks'][arm].items() if k.startswith('feedback.')})
            checks={}
            for level,channels,size in [('level2',96,32),('level3',192,16)]:
                x=torch.randn(1,channels,size,size)
                context=dict(image=image,available=arm=='R',t1=t1,features=features)
                y=module.fuse(x,level,context,polar)
                changed=dict(context,t1=t1+.1,features={k:v+.2 for k,v in features.items()})
                y2=module.fuse(x,level,changed,polar);values=module.last_fuse_values[level]
                checks[level]=dict(output_identical=bool(torch.equal(y,y2)),max_output_change=float((y-y2).abs().max()),
                    first16_zero=bool(torch.count_nonzero(values[:,16:32])==0),
                    previous6_zero=bool(torch.count_nonzero(values[:,51:57])==0),
                    output_weight_norm=float(module.inject[level].body[4].weight.norm()))
                assert checks[level]['output_weight_norm']>0
                if arm=='B':assert checks[level]['output_identical'] and checks[level]['first16_zero'] and checks[level]['previous6_zero']
                else:assert not checks[level]['output_identical'] and checks[level]['max_output_change']>0
            result[arm]=checks
    return dict(passed=True,post_hoc=True,cpu_only=True,trained_weights=True,synthetic_inputs=True,
        full_image_inference=False,not_a_replacement_for_prospective_preflight=True,checks=result)

def main():
    assert not (OUT/'POSTREVIEW_CLOSEOUT.json').exists(),'Existing completed review must not be overwritten'
    critical=[ROOT/x for x in ['weights/final.pt','weights/initial.pt','weights/last.pt','logs/training.csv','metrics/validation64.json','reports/summary.json','config/locked.json']]
    before={str(p.relative_to(ROOT)):sha(p) for p in critical}
    original=read(ROOT/'artifact_manifest.json');bad=[]
    for record in original['files']:
        p=ROOT/record['path']
        if not p.is_file() or p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:bad.append(record['path'])
    assert set(bad)=={'reports/archive_audit.json','reports/final_review_closeout.json'},bad
    scientific=audit.audit(ROOT)
    assert scientific['passed']
    local=state_checks()
    vgg=ROOT/'experiments/pretrained_models/vgg19-dcbb9e9d.pth'
    vgg_sha=sha(vgg)
    assert vgg_sha=='dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44'
    after={str(p.relative_to(ROOT)):sha(p) for p in critical};assert before==after
    review=dict(time_unix=time.time(),protocol_id=scientific['protocol_id'],scientific_id=scientific['scientific_id'],
        completed_cpu_review=True,original_archive_currently_valid=False,original_archive_mismatches=bad,
        original_manifest_sha256=sha(ROOT/'artifact_manifest.json'),original_files_preserved=True,
        scientific_audit_rerun_passed=True,steps=3000,validation=64,test=0,training_or_image_evaluation_rerun=False,
        critical_before_after_sha256=before,state_interface_supplement=local,
        vgg_dependency_post_hoc=dict(path=str(vgg),target=str(vgg.resolve()),bytes=vgg.stat().st_size,sha256=vgg_sha),
        report=scientific['report'],promotion_pass=scientific['report']['promotion_pass'],
        defects=['Old archive receipts were included and later overwritten; original archive audit was stale.',
                 'Original nonzero-interface state preflight overstated its coverage; local trained CPU adapter evidence is supplementary, not prospective full-network proof.'],
        terminal_decision='Stop fixed instance; no full872/188 promotion, no new training/hyperparameter search',
        limitations=['Single training seed','Fixed64 exploratory validation','No test inference','Repeated development validation','No full-network image sensitivity check in post-hoc CPU supplement'])
    write_new(OUT/'REVIEW.json',review)
    paths={ROOT/r['path'] for r in original['files']}
    paths.update([ROOT/'artifact_manifest.json',ROOT/'closeout.json',ROOT/'status.json'])
    paths.update(p for p in OUT.rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts)
    records=[receipt(p) for p in sorted(paths)]
    manifest=OUT/'SUPPLEMENTARY_ARTIFACT_MANIFEST.json'
    write_new(manifest,dict(protocol_id=scientific['protocol_id'],supplementary_only=True,original_archive_not_overwritten=True,
        files=records,excludes_own_manifest_and_postreview_closeout=True))
    for record in records:
        p=ROOT/record['path'];assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256'],record['path']
    final=dict(completed_cpu_review=True,scientific_evidence_valid_for_exploratory_negative_result=True,
        original_archive_valid=False,supplementary_archive_verified=True,files=len(records),
        supplementary_manifest_sha256=sha(manifest),review_sha256=sha(OUT/'REVIEW.json'),
        protocol_id=scientific['protocol_id'],steps=3000,validation=64,test=0,promotion_pass=False,
        new_training_started=False,completed_unix=time.time())
    write_new(OUT/'POSTREVIEW_CLOSEOUT.json',final)
    print(json.dumps(final,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
