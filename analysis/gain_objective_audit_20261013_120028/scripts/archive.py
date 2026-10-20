"""CPU-only provenance archive; an unfinished run is explicitly incomplete."""
import hashlib,json,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for value in iter(lambda:f.read(8*1024*1024),b''):h.update(value)
    return h.hexdigest()
def archive():
    status=json.loads((ROOT/'status.json').read_text());complete=status['stage']=='completed'
    lines=['# 第22项：固定候选收益诊断与条件性判断短训练','',f"状态：{status['stage']}",'','复用第21组最终权重，T1、候选、Q和偏振任务网络全部冻结。先按训练内隔离角色检查收益可预测性；只有预定条件成立才训练判断器3000步。主要改动是保留浮点修改及区域内增量统计；动作、损失、学习率和安全校准规则不变。GT不进入实际前向。','']
    for split,count in [('validation',872),('test',188)]:
        p=ROOT/f'downstream/metrics/final_{split}.json'
        if not p.exists():complete=False;lines.append(f'{split} 全集评价未完成。');continue
        x=json.loads(p.read_text());complete &= x['frames']==count
        lines.append(f"{split} {x['frames']}张：T1 {x['round1']['image_mean_psnr']:.6f}dB，T2 {x['round2']['image_mean_psnr']:.6f}dB，差值 {x['iteration_gain']['image_mean_psnr']:.6f}dB；SSIM变化 {x['iteration_gain']['image_mean_ssim']:.8f}。")
    complete &= (ROOT/'downstream/metrics/mechanism_validation.json').exists()
    diagnostic_only=status['stage']=='diagnostic_complete_no_training'
    lines += ['',f'正式训练与完整实验完成：{complete}。固定候选诊断完成且未启动正式训练：{diagnostic_only}。方法是否有效须根据完整逐图、场景、结构指标、动作风险和覆盖率判断。','单seed，无匹配判断器续训对照。训练内部保留场景已被继承权重见过，不能冒充完全独立数据。历史测试有多轮探索背景，本组测试不用于调参、权重选择或追加训练。','GT理想选择及强制覆盖探针仅为诊断，不是可部署方法成绩。']
    (ROOT/'RESULTS_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    files=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.name in ('archive_manifest.json','archive_ready.json') or path.name.endswith(('.tmp','.tar.gz')) or '__pycache__' in path.parts or 'runtime' in path.relative_to(ROOT).parts:continue
        files.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)))
    (ROOT/'archive_manifest.json').write_text(json.dumps(dict(created_unix=time.time(),files=files),indent=2))
    tmp=ROOT/'artifacts.tar.gz.tmp'
    with tarfile.open(tmp,'w:gz',compresslevel=1) as tar:
        for item in files:tar.add(ROOT/item['path'],arcname=item['path'])
        tar.add(ROOT/'archive_manifest.json',arcname='archive_manifest.json')
    final=ROOT/'artifacts.tar.gz';tmp.replace(final)
    (ROOT/'archive_ready.json').write_text(json.dumps(dict(sha256=sha(final),bytes=final.stat().st_size,completed=complete or diagnostic_only,formal_experiment_completed=complete,diagnostic_only=diagnostic_only)))
    print(json.dumps(dict(archived=True,complete=complete,files=len(files))),flush=True)
if __name__=='__main__':archive()
