"""CPU-only archive/report; also runs independently after the hard GPU cutoff."""
import hashlib,json,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def archive():
    status=json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {}
    lines=['# 两轮残差纠错与直接特征反馈：唯一最新训练组','',f"状态：{status.get('stage','unknown')}",
           '','仅运行最新一组，单 seed 2026。没有匹配对照，不能将与历史结果的差异归因于新增模块。',
           '复用已完成并核验的PIDSR纠偏预训练；其评价记录来自前一实验。PolarFree直接从官方权重开始。新增跨轮多尺度任务/偏振特征融合、可访问原始观测的纠错状态以及专用RGB残差头，T2=T1+学习到的增量。不是所有输出层共享。新预算独立于已结束的旧实验，不能将与历史结果的差异直接归因于某一接口。',
           'AoLP 周期距离、有效区域掩码、正交关系目标残差是本课题适配，不能声称严格复现完整 PIDSR 方法。','']
    p=ROOT/'pretraining/metrics/final_test.json'
    if p.exists():
        x=json.loads(p.read_text());lines += [f"复用的预训练已有完整 30 测试场景，选定步数 {x['step']}。",f"四角度平均 PSNR：退化输入 {x['input_psnr']:.6f}，增强输出 {x['psnr']:.6f} dB。",f"SSIM：输入 {x['input_ssim']:.6f}，输出 {x['ssim']:.6f}。",'']
    else:lines+=['预训练完整测试未完成。','']
    p=ROOT/'pretraining/convergence.json'
    if p.exists():
        x=json.loads(p.read_text());lines += [f"预训练验证稳定性检查：{x.get('stable',False)}；实际检查步数 {x.get('step')}。此规则是预先规定的验证平台判据，不能证明全局收敛。",'']
    p=ROOT/'config/locked_run.json'
    if p.exists():
        cfg=json.loads(p.read_text());lines += [f"PolarFree 初始化：{cfg.get('task_initialization',{}).get('kind','historical_joint')}，源 SHA256 {cfg.get('source_sha256')}。",'']
    probes=sorted((ROOT/'downstream/metrics').glob('retention_step_*.json'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
    if probes:
        x=json.loads(probes[-1].read_text());lines += [f"固定验证探针（原始偏振绕过增强器）步数 {x['step']}，PSNR相对官方起点变化 {x.get('psnr_change_from_official',0):.6f} dB，主网相对参数L2变化 {x['net_g_relative_l2_drift']:.6e}。该探针只监测性能保留，不是匹配对照，也不选择权重。",'']
    complete=True
    for split,n in [('validation',872),('test',188)]:
        p=ROOT/f'downstream/metrics/final_{split}.json'
        if p.exists():
            x=json.loads(p.read_text());complete &= x['frames']==n
            a,b=x['round1'],x['round2'];lines += [f"{split}：{x['frames']} 张，步数 {x['checkpoint_step']}；T1/T2 PSNR {a['image_mean_psnr']:.6f}/{b['image_mean_psnr']:.6f} dB，SSIM {a['image_mean_ssim']:.6f}/{b['image_mean_ssim']:.6f}。",f"置信—局部误差空间 Spearman 逐图均值：{x['confidence']['spearman_image_mean']}。",'']
        else:complete=False;lines+=[f'{split} 全集评价未完成。','']
    lines += [f'实现与实验完成：{complete}。方法有效性须结合偏振误差、T1/T2、场景分布判断，不能从训练结束直接推断。',
              '历史测试集已有多轮探索背景；当前测试不用于选择权重、调参或启动追加实验。',
              '完整逐图、场景指标、置信排序与区间误差见各 metrics JSON。','',
              '参考：[PIDSR CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_PIDSR_Complementary_Polarized_Image_Demosaicing_and_Super-Resolution_CVPR_2025_paper.html)、[SGSC IJCV2025](https://arxiv.org/html/2402.18134v2)、[UniRestore CVPR2025](https://arxiv.org/html/2501.13134)。这些文献不直接证明本组合的去反射收益。']
    (ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    included=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p.name.endswith(('.tmp','.tar.gz')) or '__pycache__' in p.parts or 'runtime' in p.relative_to(ROOT).parts:continue
        if p.name=='archive_manifest.json':continue
        included.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    (ROOT/'archive_manifest.json').write_text(json.dumps(dict(created_unix=time.time(),files=included),indent=2))
    path=ROOT/'artifacts.tar.gz';tmp=ROOT/'artifacts.tar.gz.tmp'
    with tarfile.open(tmp,'w:gz',compresslevel=1) as tar:
        for r in included:tar.add(ROOT/r['path'],arcname=r['path'])
        tar.add(ROOT/'archive_manifest.json',arcname='archive_manifest.json')
    tmp.replace(path)
    (ROOT/'archive_ready.json').write_text(json.dumps(dict(sha256=sha(path),bytes=path.stat().st_size,completed=complete)))
    print(json.dumps(dict(archived=True,complete=complete,bytes=path.stat().st_size)),flush=True)
if __name__=='__main__':archive()
