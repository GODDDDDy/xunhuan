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
    lines=['# 新偏振增强器：PIDSR 纠偏预训练与置信反馈','',f"状态：{status.get('stage','unknown')}",
           '','仅运行最新一组，单 seed 2026。没有匹配对照，不能将与历史结果的差异归因于新增模块。',
           '预训练采用 PIDSR 数据、CPFA/Bayer 退化及偏振损失组，训练增强器的同分辨率恢复分支；不训练超分分支、不输入干净 RGB 引导。',
           'AoLP 周期距离、有效区域掩码、正交关系目标残差是本课题适配，不能声称严格复现完整 PIDSR 方法。','']
    p=ROOT/'pretraining/metrics/final_test.json'
    if p.exists():
        x=json.loads(p.read_text());lines += [f"预训练完整 30 测试场景，选定步数 {x['step']}。",f"四角度平均 PSNR：退化输入 {x['input_psnr']:.6f}，增强输出 {x['psnr']:.6f} dB。",f"SSIM：输入 {x['input_ssim']:.6f}，输出 {x['ssim']:.6f}。",'']
    else:lines+=['预训练完整测试未完成。','']
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
