"""Run startup integrity, continuous training, full validation and full test."""
import csv
import gc
import json
from pathlib import Path
import sys
import time
import traceback
import torch
import yaml
import run_direct_t1

ROOT = Path(__file__).resolve().parents[1]


def write_status(stage, **extra):
    data = {'stage':stage, 'time_unix':time.time(), **extra}
    tmp = ROOT/'status.json.tmp'
    tmp.write_text(json.dumps(data,indent=2))
    tmp.replace(ROOT/'status.json')
    print(json.dumps(data),flush=True)


def write_report(metrics, cfg):
    old=json.loads(Path(cfg['study']['source_K1_validation']).read_text())
    old_values={k:old[k] for k in ['image_mean_psnr','scene_macro_psnr','image_mean_ssim','scene_macro_ssim']}
    rows=['# 直接T1图像反馈：K1权重起步的两轮联合训练', '',
          '单seed2026；继承K1 aligned step3000的完整R2P、投影头与net_g。新增10000步，fresh Adam；不加入置信度。', '',
          '|输出|图像PSNR|场景PSNR|图像SSIM|场景SSIM|', '|---|---:|---:|---:|---:|']
    for label, v in [('源K1（新增训练前）',old_values),('新模型第一轮T1',metrics['round1']),('新模型第二轮T2',metrics['round2'])]:
        rows.append(f"|{label}|{v['image_mean_psnr']:.8f}|{v['scene_macro_psnr']:.8f}|{v['image_mean_ssim']:.8f}|{v['scene_macro_ssim']:.8f}|")
    rows += ['', '## 比较范围', '',
             'T2−T1是同一两轮检查点内的迭代差；新模型−源K1同时包含追加训练和结构改变，不能只归因于T1反馈。没有同额外10000步的独立K1对照，不宣称已定位循环的独立贡献。', '',
             '此文件先记录完整872张验证集评价；随后必须完成188张测试集并生成合并报告，实验才算完成。原混合RGB是两次去反射的固定输入。第二轮增强条件为T1截断到[0,1]，保留反馈梯度；对齐仅第一轮，扩散先验权重冻结。', '',
             '逐图原值、场景聚合、随机流、源检查点哈希、预检与冻结检查记录在run/和protocol_lock.json。']
    (ROOT/'analysis').mkdir(exist_ok=True)
    (ROOT/'analysis/DIRECT_T1_JOINT_RESULTS_ZH_20261008.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')


def main():
    start=time.time()
    cfg=yaml.safe_load((ROOT/'config/study.yaml').read_text())
    base=run_direct_t1.configure(cfg)
    base.add_source_path(cfg)
    device=torch.device('cuda:0')
    original_validate=base.validate
    def observed_validate(model,task,loader,device,seed):
        total=len(loader.dataset)
        if total!=872:
            return original_validate(model,task,loader,device,seed)
        started=time.time()
        def observed_batches():
            for i,batch in enumerate(loader,1):
                yield batch
                if i%20==0 or i==total:
                    base.atomic_json(ROOT/'evaluation_progress.json',
                        {'completed_frames':i,'total_frames':total,'elapsed_seconds':time.time()-started})
        return original_validate(model,task,observed_batches(),device,seed)
    base.validate=observed_validate
    base.atomic_json(ROOT/'runtime.json',{'python':sys.version,'torch':torch.__version__,
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
        'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32})
    write_status('startup_integrity')
    preflight=base.run_preflight(cfg,device)
    assert preflight['passed']
    write_status('training',additional_step=0,target_steps=10000)
    base.run_training(cfg,device,200,False)
    gc.collect();torch.cuda.empty_cache()
    ck=torch.load(ROOT/'run/checkpoints/last.pt',map_location='cpu',weights_only=False)
    assert ck['step']==200 and ck['image_feedback_enabled'] and ck['actual_rounds']==2
    adam_steps=[float(s['step']) for s in ck['optimizer']['state'].values()]
    assert adam_steps and min(adam_steps)==200 and max(adam_steps)==200
    base.atomic_json(ROOT/'run/diagnostics/stage200_continuity.json',{'passed':True,'adam_states':len(adam_steps),
        'all_adam_steps':200,'source_K1_sha256':cfg['study']['source_K1_sha256']})
    del ck
    write_status('training',additional_step=200,target_steps=10000)
    base.run_training(cfg,device,10000,True)
    gc.collect();torch.cuda.empty_cache()
    logs=list(csv.DictReader((ROOT/'run/logs/train.csv').open()))
    assert [int(r['step']) for r in logs]==list(range(1,10001))
    base.atomic_json(ROOT/'run/diagnostics/training_continuity.json',{'passed':True,'continuous_additional_steps':10000})
    write_status('full_validation',additional_step=10000,target_frames=872)
    metrics=base.run_evaluation(cfg,device,'step_10000','validation')
    assert metrics['frames']==872 and len(metrics['per_image'])==872
    metrics.update(experiment=cfg['study']['experiment'],source_K1_sha256=cfg['study']['source_K1_sha256'],
        source_training_steps=3000,additional_training_steps=10000,total_training_steps=13000,
        confidence_enabled=False,image_feedback_enabled=True,elapsed_seconds=time.time()-start)
    base.atomic_json(ROOT/'metrics/results.json',metrics)
    write_report(metrics,cfg)
    from test_completion import complete_test
    base.validate=original_validate
    write_status('full_test',additional_step=10000,target_frames=188)
    test=complete_test(base,cfg,ROOT,publish_main_status=False)
    write_status('completed',additional_step=10000,validation_frames=872,
        test_frames=188,test_required=True,test_completed=True,test_round1=test['round1'],test_round2=test['round2'],
        test_iteration_gain=test['iteration_gain'],checkpoint_sha256=test['checkpoint_sha256'],
        round1=metrics['round1'],round2=metrics['round2'],iteration_gain=metrics['iteration_gain'],
        elapsed_seconds=time.time()-start)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        write_status('failed',error=repr(exc),traceback=traceback.format_exc())
        raise
