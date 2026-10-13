"""Render the 16 predeclared scene examples; no inference or image selection."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/downstream/metrics'
OUT=ROOT/'interpretation/region_examples'
OUT.mkdir(parents=True,exist_ok=True)
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams['font.family']=font.get_name()
plt.rcParams['axes.unicode_minus']=False
records=[]
for split in ('validation','test'):
    for path in sorted((ART/'acceptance_examples'/split).glob('scene_*.npz')):
        with np.load(path,allow_pickle=False) as a, np.load(ART/'confidence_examples'/split/path.name,allow_pickle=False) as c:
            sample=str(a['sample_id']);assert sample==str(c['sample_id'])
            layers=[(c['confidence'],'第一轮质量分数 C','viridis',0,1),
                    (a['error1'],'第一轮局部误差','magma',0,.1),
                    (a['need'],'需要继续修的预测分数','viridis',0,1),
                    (a['mask'],'最终更新权重 M','viridis',0,1),
                    (a['proposal_error']-a['error1'],'未门控候选：误差变化','coolwarm',-.02,.02),
                    (a['final_error']-a['error1'],'门控结果：误差变化','coolwarm',-.02,.02)]
            fig,axes=plt.subplots(2,3,figsize=(11.8,7.3),layout='constrained')
            for ax,(values,title,cmap,lo,hi) in zip(axes.flat,layers):
                im=ax.imshow(values,cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest')
                ax.set_title(title,fontproperties=font,fontsize=11);ax.set_axis_off()
                fig.colorbar(im,ax=ax,shrink=.78,fraction=.045,pad=.025)
            fig.suptitle(f'{sample}｜固定场景首张示例\nM均值 {float(a["mask"].mean()):.5f}，零更新小区域 {float((a["mask"]==0).mean()):.1%}',fontproperties=font,fontsize=12)
            fig.supxlabel('红色：误差增加；蓝色：误差减小。候选与最终结果使用相同色条；C不是正确概率。\n这些是4×4小区域数值图，不能代替原图视觉质量判断。',fontproperties=font,fontsize=10)
            target=OUT/f'{split}_{path.stem}.png';fig.savefig(target,dpi=140);plt.close(fig)
            records.append(dict(sample_id=sample,path=target.relative_to(ROOT).as_posix(),
                selection='first recorded image per scene, fixed before evaluation',
                mask_mean=float(a['mask'].mean()),protected_cell_fraction=float((a['mask']==0).mean())))
assert len(records)==16
(OUT/'index.json').write_text(json.dumps(dict(gpu_used=False,examples=records),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(rendered=len(records),gpu_used=False)))
