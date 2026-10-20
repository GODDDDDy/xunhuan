"""CPU-only replotting of archived results. No model fitting, inference or selection."""
import os,sys,json,csv,hashlib,shutil
from pathlib import Path
W=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(W/'build/mpl_cache')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.path.insert(0,r'F:\WORK\tools\scientific_plot_py312')
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
P=W.parents[1]
A=P/'_dl/polar_gain_judgement_short_20261012/artifacts'
F=W/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white','axes.titleweight':'bold'})
BLUE='#26748B';RED='#B94F43';GOLD='#C58B2A';GREY='#6B7280'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(fig,name):
 for ext in ('png','svg','pdf'):fig.savefig(F/(name+'.'+ext),dpi=200,bbox_inches='tight')
 plt.close(fig)
def ranks(a):
 _,inv,c=np.unique(a,return_inverse=True,return_counts=True);return (np.cumsum(c)-c+(c-1)/2+1)[inv]
def rho(a,b):return float(np.corrcoef(ranks(a),ranks(b))[0,1])
val=read(A/'downstream/metrics/final_validation.json');test=read(A/'downstream/metrics/final_test.json')
cal=read(A/'downstream/config/action_calibration.json');summary=read(P/'_dl/polar_gain_judgement_short_20261012/interpretation/result_summary.json')
audit=read(W/'data/cpu_structure_audit.json')
with (A/'config/manifest.csv').open(encoding='utf-8-sig') as f:manifest={r['sample_id']:r for r in csv.DictReader(f)}
ids={r['sample_id']:i for i,r in enumerate(val['per_image'])}
cases={}
for scene in ('01','30','55'):
 with np.load(A/f'downstream/metrics/confidence_examples/validation/scene_{scene}.npz') as z: q={k:z[k].copy() for k in z.files}
 sample=q['sample_id'].item();idx=ids[sample]
 with np.load(A/f'diagnostics/validation_action_maps/{idx:04d}.npz') as z: d={k:z[k].copy() for k in z.files}
 assert d['sample_id'].item()==sample
 row=val['per_image'][idx]; cost=d['actual_action_cost'];scale=cal['cost_scale']
 imgs={n:np.array(Image.open(A/f'downstream/examples/validation/{scene}_{n}.png')) for n in ('T1','candidate','GT','T2')}
 assert np.array_equal(imgs['T1'],imgs['T2'])
 c=q['confidence'];e=q['error1'];gain=-cost[2];oracle=np.argmin(np.concatenate([np.zeros_like(cost[:1]),cost],0),0)
 cases[scene]=dict(sample_id=sample,source_index=idx,shape=list(imgs['T1'].shape),metrics=row,quality_error_rho=rho(c.ravel(),e.ravel()),quality_full_action_gain_rho=rho(c.ravel(),gain.ravel()),beneficial_full_action_fraction=float((gain>.02*scale).mean()),harmful_full_action_fraction=float((gain<-.02*scale).mean()),oracle_nonzero_fraction=float((oracle>0).mean()),actual_deployed_coverage=0.,map_source=str(A/f'diagnostics/validation_action_maps/{idx:04d}.npz'),image_source=str(A/f'downstream/examples/validation/{scene}_T1.png'),dataset_manifest=manifest[sample])
 if scene!='30':continue
 panels=F/'panels';panels.mkdir(exist_ok=True)
 for n,img in imgs.items():Image.fromarray(img).save(panels/f'30_{n}.png')
 color_limits=dict(C=[0,1],error=[0,float(np.quantile(e,.98))],gain=[-float(np.quantile(np.abs(gain),.98)),float(np.quantile(np.abs(gain),.98))])
 for name,array,cmap in [('C',c,'viridis'),('error',e,'magma'),('gain',gain,'RdBu')]:
  lo,hi=color_limits[name];v=np.clip((array-lo)/(hi-lo),0,1);Image.fromarray((matplotlib.colormaps[cmap](v)[...,:3]*255).astype('uint8')).save(panels/f'30_{name}.png')
  Image.fromarray((matplotlib.colormaps[cmap](np.tile(np.linspace(0,1,500),(22,1)))[...,:3]*255).astype('uint8')).save(panels/f'30_{name}_legend.png')
 colors=np.array([[236,239,241],[38,116,139],[197,139,42],[185,79,67]],np.uint8)
 Image.fromarray(colors[oracle]).save(panels/'30_oracle.png');Image.fromarray(colors[np.zeros_like(oracle)]).save(panels/'30_action.png')
 cases[scene]['color_limits']=color_limits
 fig,ax=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
 ax[0].imshow(imgs['T1']);ax[0].set_title('第一轮 T1（仅显示用8位图）')
 im=ax[1].imshow(c,cmap='viridis',vmin=0,vmax=1);ax[1].set_title('质量分数 C\n分数高：预测误差较低');fig.colorbar(im,ax=ax[1],orientation='horizontal',fraction=.06)
 vmax=float(np.quantile(e,.98));im=ax[2].imshow(e,cmap='magma',vmin=0,vmax=vmax);ax[2].set_title('GT核验的第一轮局部MAE\n色标在98%分位截断');fig.colorbar(im,ax=ax[2],orientation='horizontal',fraction=.06)
 for a in ax:a.set_axis_off()
 fig.suptitle(f'{sample}  |  C与局部误差 Spearman = {cases[scene]["quality_error_rho"]:.3f}\n置信图来自浮点归档，误差使用GT计算；C不是正确概率',fontsize=13)
 save(fig,'01_quality_case30')
 fig,ax=plt.subplots(2,3,figsize=(14,8.4),layout='constrained')
 for a,n,title in zip(ax[0],['T1','candidate','GT'],['第一轮 T1','第二轮候选 U（门控前）','数据集真值 GT']):a.imshow(imgs[n]);a.set_title(title)
 lim=float(np.quantile(np.abs(gain),.98));norm=TwoSlopeNorm(0,-lim,lim)
 im=ax[1,0].imshow(gain,cmap='RdBu',norm=norm);ax[1,0].set_title('候选1倍修正的真实MSE收益\n蓝：更好，红：更坏');fig.colorbar(im,ax=ax[1,0],orientation='horizontal',fraction=.06)
 im=ax[1,1].imshow(np.zeros_like(oracle),cmap='Greys',vmin=0,vmax=1);ax[1,1].set_title('实际部署动作 M\n全部拒绝，最终T2=T1')
 im=ax[1,2].imshow(oracle,cmap=matplotlib.colors.ListedColormap(['#ECEFF1',BLUE,GOLD,RED]),norm=matplotlib.colors.BoundaryNorm([-.5,.5,1.5,2.5,3.5],4),interpolation='nearest');ax[1,2].set_title('GT理想动作（不可部署）\n0 / 0.25 / 0.5 / 1');cb=fig.colorbar(im,ax=ax[1,2],orientation='horizontal',fraction=.06,ticks=[0,1,2,3]);cb.ax.set_xticklabels(['0','.25','.5','1'])
 for a in ax.flat:a.set_axis_off()
 fig.suptitle(f'{sample} | 候选与T1同图比较；热图基于原始浮点4×4实际成本\n图像仅显示时裁剪至[0,1]；指标不由PNG重算；真值选择仅诊断',fontsize=12)
 save(fig,'02_candidate_and_acceptance_case30')
 # Post-hoc illustrative extremes, not representative effect estimates or model selection.
 side=32;h,w=gain.shape;hh=h//side;ww=w//side
 block=gain[:hh*side,:ww*side].reshape(hh,side,ww,side).mean((1,3))
 luma=(imgs['GT'][:hh*side*4,:ww*side*4,:3]/255.*np.array([.299,.587,.114])).sum(2)
 texture=luma.reshape(hh,side*4,ww,side*4).std((1,3));eligible=texture>.035
 positions=[np.unravel_index(np.argmax(np.where(eligible,block,-np.inf)),block.shape),np.unravel_index(np.argmin(np.where(eligible,block,np.inf)),block.shape)]
 fig,ax=plt.subplots(2,3,figsize=(10.8,7.3),layout='constrained');roi=[]
 for k,(yy,xx) in enumerate(positions):
  x,y=int(xx*side*4),int(yy*side*4);s=side*4
  for j,n in enumerate(['T1','candidate','GT']):ax[k,j].imshow(imgs[n][y:y+s,x:x+s],interpolation='nearest');ax[k,j].set_axis_off();ax[k,j].set_title(n if k==0 else '')
  for n in ['T1','candidate','GT']:Image.fromarray(imgs[n][y:y+s,x:x+s]).save(panels/f'30_roi{k}_{n}.png')
  g=float(block[yy,xx]);roi.append(dict(x=x,y=y,width=s,height=s,mean_raw_mse_gain=g,gt_luma_std=float(texture[yy,xx]),selection='post-hoc max/min 128x128 patch gain among GT luminance std>.035; illustration only'))
  ax[k,0].set_title(('有益' if k==0 else '有害')+f'区域的T1\n原始MSE收益={g:+.2e}',fontsize=11)
  ax[k,0].set_title(('有益' if k==0 else '有害')+f'区域的T1：({x},{y})\n128×128，原始MSE收益={g:+.2e}',fontsize=11)
 fig.suptitle('同一张图里，第二轮能修好一些区域，也会改坏另一些区域\n事后选择有纹理的128×128正/负极值区域，仅解释现象',fontsize=12);save(fig,'03_good_bad_roi_case30');cases[scene]['illustrative_rois']=roi
 fig,ax=plt.subplots(figsize=(7,6.2),layout='constrained');ax.imshow(imgs['T1']);ax.set_axis_off()
 from matplotlib.patches import Rectangle
 for k,r in enumerate(roi):
  color=BLUE if k==0 else RED;ax.add_patch(Rectangle((r['x'],r['y']),r['width'],r['height'],fill=False,edgecolor=color,lw=2.5));ax.text(r['x'],max(0,r['y']-8),'有益裁剪' if k==0 else '有害裁剪',color='white',fontsize=10,bbox=dict(facecolor=color,edgecolor='none',pad=2))
 ax.set_title('场景30：裁剪的原图位置（T1）');save(fig,'12_roi_locations')
 fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');stride=max(c.size//8000,1);cc=c.ravel()[::stride];ee=e.ravel()[::stride];gg=gain.ravel()[::stride]
 im=ax[0].hexbin(cc,ee,gridsize=45,mincnt=1,bins='log',cmap='Blues');ax[0].set(xlabel='质量分数 C',ylabel='T1局部MAE',title=f'质量排序：ρ={cases[scene]["quality_error_rho"]:.3f}')
 im=ax[1].hexbin(cc,gg,gridsize=45,mincnt=1,bins='log',cmap='Blues');ax[1].axhline(0,color=RED,lw=1);ax[1].set(xlabel='质量分数 C',ylabel='候选1倍修改的实际MSE收益',title=f'修改收益：ρ={cases[scene]["quality_full_action_gain_rho"]:.3f}')
 fig.suptitle(f'{sample}：知道哪里差，不等于知道给定修正往哪个方向改\nρ为该图完整网格；散点固定步长展示，无显著性/因果结论',fontsize=12);save(fig,'04_quality_vs_benefit')
grid=[r for r in cal['grid'] if r['coverage']>0];x=[r['coverage']*100 for r in grid]
fig,ax=plt.subplots(1,2,figsize=(11.5,4.2),layout='constrained')
ax[0].plot(x,[r['accepted_harmful_rate']*100 for r in grid],'o-',color=RED);ax[0].axhline(20,color=GREY,ls='--');ax[0].set(xscale='log',xlabel='接受区域比例 (%)，对数轴',ylabel='接受中明显有害比例 (%)',title='更严格的门槛没有带来可靠修改')
for i,r in enumerate(grid):ax[0].annotate(str(r['margin']),(x[i],r['accepted_harmful_rate']*100),xytext=(4,5),textcoords='offset points',fontsize=8)
ax[1].plot(x,[r['mean_mse_improvement']*1e6 for r in grid],'o-',color=BLUE);ax[1].axhline(0,color=GREY,ls='--');ax[1].set(xscale='log',xlabel='接受区域比例 (%)，对数轴',ylabel='全图归一MSE净改善 (x 1e-6)',title='所有非零覆盖候选门槛均不合格')
fig.suptitle('训练内部安全校准：20%风险线；门槛1.6时零覆盖\n空接受集不等于判断准确，未根据验证/测试放松规则',fontsize=12);save(fig,'05_risk_coverage')
fig,ax=plt.subplots(figsize=(8.8,4.1),layout='constrained');x=np.arange(3);ax.bar(x-.17,[-.0340528734072,-.0454988252519,-.0551101210951],.34,color=BLUE,label='872验证');ax.bar(x+.17,[.0228310487244,.018396649371,.0028760694289],.34,color=GOLD,label='188历史测试');ax.axhline(0,color=GREY,lw=1);ax.set(xticks=x,xticklabels=['0.25倍修正','0.5倍修正','1倍修正'],ylabel='收益排序Spearman（逐图平均）',title='最终判断头的收益排序接近0');ax.legend(frameon=False);save(fig,'06_final_action_ranking')
fig,ax=plt.subplots(1,2,figsize=(11.5,4.2),layout='constrained');scenes=[];gp=[];gs=[]
for s in val['per_scene']:
 scenes.append(s['scene']);gp.append(s['ungated_candidate']['psnr']-s['round1']['psnr']);gs.append(s['ungated_candidate']['ssim']-s['round1']['ssim'])
ax[0].bar(scenes,gp,color=[BLUE if z>=0 else RED for z in gp]);ax[0].axhline(0,color=GREY,lw=1);ax[0].set(xlabel='验证场景',ylabel='候选−T1 PSNR (dB)',title='候选的场景收益不稳定')
ax[1].bar(scenes,np.array(gs)*1000,color=[BLUE if z>=0 else RED for z in gs]);ax[1].axhline(0,color=GREY,lw=1);ax[1].set(xlabel='验证场景',ylabel='候选−T1 SSIM (x 1e-3)',title='像素误差和结构指标并不一致');save(fig,'07_scene_candidate_gains')
with np.load(A/'diagnostics/feature_cache/safety.npz') as z:xx=z['x'];yy=z['y']
mean_sq=(xx[:,9:12]**2).sum(1);energy=xx[:,43:46].sum(1);ratio=mean_sq/np.maximum(energy,1e-15)
stds=xx.std(0);group_stats={label:dict(std_median=float(np.median(stds[cols])),mean_abs=float(np.abs(xx[:,cols]).mean())) for label,cols in [('RGB',range(3)),('T1',range(3,6)),('mean_delta',range(9,12)),('polar',range(12,17)),('error_C',range(17,19)),('projected_task',range(19,27)),('state',range(27,43)),('delta_energy',range(43,46)),('abs_delta',range(46,49)),('cancellation_ratio',[49])]}
fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');ax.bar(list(group_stats),[v['std_median'] for v in group_stats.values()],color=BLUE);ax.set_yscale('log');ax.tick_params(axis='x',rotation=30,labelsize=9);ax.set(ylabel='通道标准差中位数（对数轴）',title='正式网络直接输入不同量级通道，诊断探针则先标准化');save(fig,'11_input_channel_scales')
fig,ax=plt.subplots(figsize=(8.5,4.1),layout='constrained');ax.hist(ratio,bins=np.linspace(0,1,41),color=BLUE);ax.set(xlabel='(区域平均增量)² / 区域平均(增量²)，RGB求和',ylabel='固定安全裁剪采样区域数',title='区域平均确实会抵消正负修改');ax.text(.96,.9,f'中位比例 {np.median(ratio):.3f}',ha='right',transform=ax.transAxes);save(fig,'08_pooling_information')
fig,ax=plt.subplots(figsize=(9,4.4),layout='constrained');names=['实际最终T2','未门控候选','GT选择 4×4','GT选择 16×16','GT选择 32×32'];g=[0,-.0172327300760422,.25618506178595,.20829251107787,.18655843962825];ax.bar(names,g,color=[GREY,RED,GOLD,GOLD,GOLD]);ax.axhline(0,color=GREY,lw=1);ax.set(ylabel='872验证平均PSNR增量 (dB)',title='局部潜力存在，但真值选择无法部署');save(fig,'09_oracle_deployment_gap')
monitor=[]
for name in ['initial']+[f'judgement_{s}' for s in range(500,3001,500)]:
 m=read(A/f'downstream/metrics/probe_{name}.json');monitor.append((0 if name=='initial' else int(name.split('_')[-1]),m['mean_loss']))
fig,ax=plt.subplots(figsize=(8.7,4),layout='constrained');ax.plot(*zip(*monitor),'o-',color=BLUE);ax.set(xlabel='判断器训练步数',ylabel='固定32偏差裁剪动作成本损失',title='3000步内没有看到明确的持续下降');ax.ticklabel_format(style='plain',axis='y',useOffset=False);save(fig,'10_training_monitor')
(W/'data/evidence_summary.json').write_text(json.dumps(dict(cases=cases,input_group_scales=group_stats,pooling_ratio_median=float(np.median(ratio)),pooling_ratio_quantiles=np.quantile(ratio,[.1,.5,.9]).tolist(),training_monitor=monitor,calibration=cal['grid'],final_ranking=dict(validation=[-.0340528734072,-.0454988252519,-.0551101210951],test=[.0228310487244,.018396649371,.0028760694289]),full_evaluation=dict(validation=val['round1'],test=test['round1']),cpu_only=True,new_training=False,figure_sources='Frozen experiment22 archives. All heatmaps from floats; RGB PNG display only.'),ensure_ascii=False,indent=2),encoding='utf-8')
# Preserve a compact table of every actual module, including dormant modules.
with (W/'data/module_inventory.csv').open('w',newline='',encoding='utf-8-sig') as f:
 wr=csv.DictWriter(f,fieldnames=['owner','name','kind','direct_parameters']);wr.writeheader()
 for owner,rows in [('enhancer_system',audit['module_inventory']),('PolarFree_net_g',audit['task_modules'])]+[(k,v['modules']) for k,v in audit['priors'].items()]:
  for r in rows:wr.writerow(dict(owner=owner,**{k:r.get(k,0) for k in ['name','kind','direct_parameters']}))
print(json.dumps(dict(figures=10,cases=list(cases),pooling_ratio_median=float(np.median(ratio)),module_rows=sum(len(audit[k]) for k in ['module_inventory','task_modules']),new_gpu=False),ensure_ascii=False))
