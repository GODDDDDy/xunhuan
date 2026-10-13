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
    lines=['# 第21组：修复需求引导的短训练','',f"状态：{status['stage']}",'','复用第20组最终权重；第一轮全部固定。新增预测修复需求Q，用于第二轮状态条件；GT只造监督标签，不进入前向。候选2000步，随后固定候选及Q训练既有动作成本判断1000步。训练内场景分工与校准不使用验证/测试。原预训练及官方权重仅是来源谱系。','']
    for split,count in [('validation',872),('test',188)]:
        p=ROOT/f'downstream/metrics/final_{split}.json'
        if not p.exists():complete=False;lines.append(f'{split} 全集评价未完成。');continue
        x=json.loads(p.read_text());complete &= x['frames']==count
        lines.append(f"{split} {x['frames']}张：T1 {x['round1']['image_mean_psnr']:.6f}dB，T2 {x['round2']['image_mean_psnr']:.6f}dB，差值 {x['iteration_gain']['image_mean_psnr']:.6f}dB；SSIM变化 {x['iteration_gain']['image_mean_ssim']:.8f}。")
    complete &= (ROOT/'downstream/metrics/mechanism_validation.json').exists()
    lines += ['',f'实现与完整实验完成：{complete}。方法是否有效须根据完整逐图、场景、结构指标、动作风险和覆盖率判断。','单seed，无匹配续训对照；新增监督和训练方式同时变化。训练内部保留场景已被继承权重见过，不能冒充完全独立数据。历史测试有多轮探索背景，本组测试不用于调参、权重选择或追加训练。','使用GT的理想选择仅用于固定验证诊断，不能当作真实模型效果。']
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
    (ROOT/'archive_ready.json').write_text(json.dumps(dict(sha256=sha(final),bytes=final.stat().st_size,completed=complete)))
    print(json.dumps(dict(archived=True,complete=complete,files=len(files))),flush=True)
if __name__=='__main__':archive()
