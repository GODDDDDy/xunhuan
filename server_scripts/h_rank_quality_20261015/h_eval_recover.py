"""Evaluation-only memory-lifetime repair; no training or changed predictions.

Original locked scripts and failed attempt are preserved. Release unused E1
feature tensors before D1; retain exactly the same refined polarization tensor.
"""
import gc,json,os,sys,time
import torch
import h_run as original
from common import ROOT,seed,sha,json_save,status,guard_gpu
import study


@torch.no_grad()
def memory_first(model,task,batch):
    p0,image=batch['polar'],batch['official_rgb']
    enhanced=study.enhance(model,p0,p0,image,project=False)
    refined=enhanced.refined_polar
    del enhanced
    gc.collect()
    if p0.is_cuda:torch.cuda.empty_cache()
    t1,features=study.task_forward(model,task,refined,image,p0)
    return (image,t1,study.polarization_summary(p0),features)


def main():
    torch.set_num_threads(2)
    assert (ROOT/'weights/final_heads.pt').exists()
    budget=json.loads((ROOT/'budget.json').read_text());assert time.time()<budget['deadline_unix']
    cfg,plan,model,task,raw,source,heads=original.load(torch.device('cpu'))
    import copy
    small=copy.deepcopy(cfg);small['data']['crop_size']=32
    batch=study.base.unpack_batch(next(study.microbatches(next(iter(study.base.scheduled_loader(small,study.base.build_base_config(small),1))))),torch.device('cpu'))
    seed(111);a=original.first_inputs(model,task,batch)
    seed(111);b=memory_first(model,task,batch)
    assert all(torch.equal(a[i],b[i]) for i in range(3))
    assert all(torch.equal(a[3][k],b[3][k]) for k in a[3])
    with torch.no_grad():assert torch.equal(model.confidence_head(*a),model.confidence_head(*b))
    del model,task,raw,source,heads,a,b,batch;gc.collect()
    cfg,plan,model,task,raw,source,heads=original.load(torch.device('cuda:0'));guard_gpu()
    ck=torch.load(ROOT/'weights/final_heads.pt',map_location='cpu',weights_only=False)
    assert ck['step']==plan['steps'] and ck['protocol_id']==plan['protocol_id']
    assert sha(ROOT/'weights/final_heads.pt')==json.loads((ROOT/'diagnostics/paired_training.json').read_text())['endpoint_sha256']
    for arm,h in heads.items():h.load_state_dict(ck['heads'][arm],strict=True)
    # Check actual GPU values as well; only the lifetime of unused tensors differs.
    small=copy.deepcopy(cfg);small['data']['crop_size']=32
    batch=study.base.unpack_batch(next(study.microbatches(next(iter(study.base.scheduled_loader(small,study.base.build_base_config(small),1))))),torch.device('cuda:0'))
    seed(111);a=original.first_inputs(model,task,batch)
    seed(111);b=memory_first(model,task,batch)
    assert all(torch.equal(a[i],b[i]) for i in range(3))
    assert all(torch.equal(a[3][k],b[3][k]) for k in a[3])
    for h in heads.values():
        with torch.no_grad():assert torch.equal(h(*a),h(*b))
    original.audit(model,task,source)
    del a,b,batch,source,raw,ck;gc.collect();torch.cuda.empty_cache()
    original.first_inputs=memory_first
    json_save(ROOT/'diagnostics/evaluation_repair.json',dict(
        code_sha256=sha(__file__),original_protocol_id=plan['protocol_id'],
        original_endpoint_sha256=sha(ROOT/'weights/final_heads.pt'),
        CPU_and_GPU_small_first_T1_features_and_H_bit_exact=True,
        change='release unused E1 features before D1; CUDA allocator expandable_segments=True',
        image_resized=False,tiling=False,weights_changed=False,training_restarted=False,
        original_deadline_unix=budget['deadline_unix'],time_unix=time.time()))
    passed=original.evaluate(cfg,plan,model,task,heads)
    if passed:
        status('full_confirmation_starting',weights_locked=True,no_more_training=True)
        original.evaluate(cfg,plan,model,task,heads,split='validation')
        original.evaluate(cfg,plan,model,task,heads,split='test')
        json_save(ROOT/'full_confirmation.json',dict(completed=True,validation=872,test=188,
                  steps_per_arm=plan['steps'],weights_locked_before_test=True,T2_evaluated=False,
                  protocol_id=plan['protocol_id'],time_unix=time.time()))
        status('full_confirmation_completed',validation=872,test=188,steps_per_arm=plan['steps'])


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        json_save(ROOT/'diagnostics'/f'evaluation_repair_failure_{int(time.time())}.json',dict(traceback=traceback.format_exc(),time_unix=time.time()))
        status('evaluation_repair_failed',traceback=traceback.format_exc());raise
