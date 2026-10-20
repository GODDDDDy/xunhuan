"""CPU-only paired endpoint summary. Never selects a checkpoint."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
results={};lines=['# 真实/固定质量条件的匹配候选短训练','', '同一第21initial起点，两组各2000步、单seed2026；仅feedback与Q更新，第一轮/H/主干冻结。直接评价候选U，M不执行。不是从头初始化消融，不保证收敛或稳定。','', '|数据|T1 PSNR|真实U PSNR|固定U PSNR|真实−T1|真实−固定|','|---|---:|---:|---:|---:|---:|']
for split in ('validation','test'):
    result=json.loads((ROOT/'metrics'/f'final_{split}.json').read_text());m=result['image_means'];t=m['t1'];real=m['real_quality'];fixed=m['fixed_quality']
    lines.append(f"|{split}|{t['psnr']:.6f}|{real['psnr']:.6f}|{fixed['psnr']:.6f}|{real['psnr']-t['psnr']:+.6f}|{real['psnr']-fixed['psnr']:+.6f}|")
    differences={k:{metric:m[k][metric]-t[metric] for metric in ('psnr','ssim')} for k in ('real_quality','fixed_quality')};differences['real_minus_fixed']={metric:real[metric]-fixed[metric] for metric in ('psnr','ssim')}
    scenes=result['per_scene'];scene_diffs=[dict(scene=r['scene'],frames=r['frames'],real_minus_T1={metric:r['means']['real_quality'][metric]-r['means']['t1'][metric] for metric in ('psnr','ssim')},real_minus_fixed={metric:r['means']['real_quality'][metric]-r['means']['fixed_quality'][metric] for metric in ('psnr','ssim')}) for r in scenes]
    results[split]=dict(image_means=m,image_mean_differences=differences,scene_differences=scene_diffs,scene_macro_real_minus_fixed={metric:sum(r['real_minus_fixed'][metric] for r in scene_diffs)/len(scene_diffs) for metric in ('psnr','ssim')})
    fields=['sample_id','scene','T1_PSNR','real_U_PSNR','fixed_U_PSNR','real_minus_T1_PSNR','real_minus_fixed_PSNR','T1_SSIM','real_U_SSIM','fixed_U_SSIM']
    with (ROOT/'metrics'/f'paired_{split}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in result['rows']:
            a,b=r['arms']['real_quality'],r['arms']['fixed_quality'];w.writerow(dict(sample_id=r['sample_id'],scene=r['scene'],T1_PSNR=r['t1']['psnr'],real_U_PSNR=a['psnr'],fixed_U_PSNR=b['psnr'],real_minus_T1_PSNR=a['psnr']-r['t1']['psnr'],real_minus_fixed_PSNR=a['psnr']-b['psnr'],T1_SSIM=r['t1']['ssim'],real_U_SSIM=a['ssim'],fixed_U_SSIM=b['ssim']))
    lines += ['',f'## {split}场景差','', '|场景|真实−T1 PSNR|真实−固定 PSNR|真实−固定 SSIM|','|---|---:|---:|---:|']+[f"|{r['scene']}|{r['real_minus_T1']['psnr']:+.6f}|{r['real_minus_fixed']['psnr']:+.6f}|{r['real_minus_fixed']['ssim']:+.6f}|" for r in scene_diffs]
lines += ['', '结果只支持这个历史起点、冻结范围、训练目标、单seed与短预算下的比较。真实优于固定但仍低于T1只能称减轻退化。正均值不自动代表跨场景稳定，不将相关帧当独立重复；验证/测试未用于调模型、常数、步数或选权重。来源有历史测试探索及验证重复使用背景。两组没有普通续训以外的随机种子重复。']
(ROOT/'metrics/summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False));(ROOT/'reports/RESULT_ZH.md').write_text('\n'.join(lines)+'\n')
