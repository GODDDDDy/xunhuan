"""CPU-only descriptive analysis of eight pre-fixed floating examples; no tuning."""
import argparse,csv,hashlib,json,os
from pathlib import Path
import numpy as np
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parents[1]/'runtime_cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser();parser.add_argument('--source');parser.add_argument('--output');args=parser.parse_args()
ROOT=Path(args.output) if args.output else Path(__file__).resolve().parents[1]
SOURCE=Path(args.source) if args.source else ROOT.parents[1]/'_dl/polar_confidence_candidate_diagnostic_20261014/artifacts'
(ROOT/'figures').mkdir(parents=True,exist_ok=True)
def pool(x):
    h,w=x.shape[-2:];assert h%4==0 and w%4==0
    return x.reshape(h//4,4,w//4,4).mean((1,3))
def corr(x,y):
    # Reuse the project's NumPy average-rank definition; no new SciPy dependency.
    def rank(a):
        _,inverse,counts=np.unique(a,return_inverse=True,return_counts=True)
        ranks=np.cumsum(counts)-counts+(counts-1)/2+1
        return ranks[inverse]
    rx,ry=rank(x.ravel()),rank(y.ravel())
    r=float(np.corrcoef(rx,ry)[0,1]) if rx.std() and ry.std() else float('nan')
    return r if np.isfinite(r) else None
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
rows=[]
for p in sorted((SOURCE/'examples').glob('*.npz')):
    with np.load(p) as f:
        image,gt,t1,c=f['image'],f['target'],f['t1'],f['c_pred'][0]
        e=pool(np.abs(t1.clip(0,1)-gt).mean(0));m1=pool(((t1-gt)**2).mean(0))
        gains={k:m1-pool(((f['u_'+k]-gt)**2).mean(0)) for k in ('real_quality','fixed_quality','shifted_quality')}
        harm=np.maximum(-gains['real_quality'],0).ravel();sort=np.sort(harm)[::-1];total=float(sort.sum())
        shares={str(q):float(sort[:max(1,int(np.ceil(q*len(sort))))].sum()/total) if total else 0. for q in (.01,.05,.1,.25)}
        row=dict(sample=p.stem,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),regions=c.size,
                 rho_C_T1_MAE=corr(c,e),rho_C_real_gain=corr(c,gains['real_quality']),rho_T1_error_real_gain=corr(e,gains['real_quality']),
                 real_mean_mse_improvement=float(gains['real_quality'].mean()),fixed_mean_mse_improvement=float(gains['fixed_quality'].mean()),
                 real_improved_region_fraction=float((gains['real_quality']>0).mean()),harm_concentration=shares)
        rows.append(row)
        lim=max(float(np.quantile(np.abs(np.concatenate([v.ravel() for v in gains.values()])),.99)),1e-8)
        fig,axes=plt.subplots(2,4,figsize=(14,7),constrained_layout=True)
        for ax,a,title in zip(axes[0],(image,t1,f['u_real_quality'],gt),('Observation','Frozen T1','Candidate U: real quality','GT (evaluation only)')):
            ax.imshow(a.transpose(1,2,0).clip(0,1));ax.set_title(title);ax.axis('off')
        for ax,a,title,cmap,vmin,vmax in zip(axes[1],(c,e,gains['real_quality'],gains['fixed_quality']),('Quality C','T1 local MAE','Real: local MSE improvement','Fixed: local MSE improvement'),('viridis','magma','RdBu','RdBu'),(0,0,-lim,-lim),(1,float(np.quantile(e,.99)),lim,lim)):
            im=ax.imshow(a,cmap=cmap,vmin=vmin,vmax=vmax);ax.set_title(title);ax.axis('off');fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle(p.stem+' | 4x4 regions; positive improvement = better; display clipped, metrics use floats',fontsize=11)
        fig.savefig(ROOT/'figures'/(p.stem+'.png'));plt.close(fig)
    print(json.dumps(row),flush=True)
payload=dict(samples=rows,frames=8,source_total_metric_frames=28,training=False,gpu_used=False,GT_evaluation_only=True,thresholds_not_tuned=True,interpretation='Descriptive correlations; no independent-pixel p-values, no causal attribution. Only first fixed example per scene, not all 28 regional outputs.')
(ROOT/'region_analysis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
with (SOURCE/'metrics/paired_per_scene.csv').open(encoding='utf-8-sig') as f:scenes=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True);x=np.arange(len(scenes))
for j,(name,color,label) in enumerate((('real_quality','#355C7D','Real'),('fixed_quality','#D88A4B','Fixed'),('shifted_quality','#778D7D','Shifted'))):
    ax.bar(x+(j-1)*.24,[float(r[name+'__u_minus_t1_psnr']) for r in scenes],width=.23,color=color,label=label)
ax.axhline(0,color='#444444',linewidth=.7);ax.set_xticks(x,[r['scene'] for r in scenes]);ax.set_xlabel('Scene');ax.set_ylabel('Candidate minus T1 PSNR (dB)');ax.legend(frameon=False,ncol=3);ax.set_title('All 28 fixed images / 8 scenes; frozen-model inference intervention')
fig.savefig(ROOT/'figures/scene_gain.png');fig.savefig(ROOT/'figures/scene_gain.svg');plt.close(fig)
lines=['# 已有浮点示例的区域分析（CPU）','', '全部28张有整图指标；本次区域分析仅使用8张预先固定示例，每场景第一张，不按成绩挑图。C对应T1局部质量，局部收益为MSE(T1,GT)−MSE(U,GT)，正数表示改好。没有训练、没有GPU、不调可部署阈值。','', '|固定示例|C与T1误差排序|C与真实局部收益排序|真实候选MSE改善|最坏10%区域占全部局部损害|','|---|---:|---:|---:|---:|']
for r in rows:lines.append(f"|{r['sample']}|{r['rho_C_T1_MAE']:.4f}|{r['rho_C_real_gain']:.4f}|{r['real_mean_mse_improvement']:.8f}|{r['harm_concentration']['0.1']:.1%}|")
lines += ['', '这些相关性只描述已保存图像，不能直接定位模块责任。损害集中度按全部4×4区域中最有害的前10%计算，分母为负收益区域损害总量，不是全图净误差；其中可能同时存在正收益区域。30号场景只有easy/0000示例，因此不能代表该场景另外三张的空间分布。图像仅为显示裁到[0,1]，统计保留原始浮点。', '', '分析不改变后续预先锁定的匹配训练组、损失、2000步预算或常数校准规则。']
(ROOT/'REGION_ANALYSIS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
