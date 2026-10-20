"""Close the read-only review delivery; package figures and render a static PDF."""
import os,json,csv,hashlib,shutil,zipfile
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1'
W=Path(__file__).resolve().parents[1];P=W.parents[1];A=P/'_dl/polar_gain_judgement_short_20261012/artifacts'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
e=json.loads((W/'data/evidence_summary.json').read_text(encoding='utf-8'));audit=json.loads((W/'data/cpu_structure_audit.json').read_text(encoding='utf-8-sig'))
with (A/'config/manifest.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
groups={s:[r for r in rows if r['split']==s] for s in ('train','validation','test')}
coverage={s:dict(frames=len(r),scene_ids=len({x['scene'] for x in r}),difficulty_scene_keys=len({(x['difficulty'],x['scene']) for x in r}),dataset_directory=sorted({x['official_rgb'].split('/polargb/')[-1].split('/')[0] for x in r})) for s,r in groups.items()}
assert not ({r['scene'] for r in groups['train']}&{r['scene'] for r in groups['validation']})
data_audit=dict(current_manifest=coverage,train_validation_scene_disjoint=True,author_paper_v1_training_images=6312,author_paper_v1_training_scenes=56,author_paper_v1_testing_images=188,author_paper_v1_testing_scenes=11,author_source='https://arxiv.org/html/2503.18055v1#S5.SS1',official_checkpoint_exact_training_manifest_available=False,validation_directory_is_original_training_pool=True,interpretation='No current train/validation scene overlap. Official pretraining exposure to validation must be audited; paper and released scene mappings differ. No proven runtime GT leak.')
(W/'data/data_split_exposure_audit.json').write_text(json.dumps(data_audit,ensure_ascii=False,indent=2),encoding='utf-8')
report=W/'output/当前网络技术审查.md';t=report.read_text(encoding='utf-8')
addition='''## 补查：运行时GT隔离与预训练数据独立性不同

当前manifest的训练5440张、验证872张均来自数据的 `train` 目录，两者场景编号无交叉（51个训练、8个验证）；测试188张来自 `test`。我们自己的场景划分与GT隔离检查通过。**但这不等于官方预训练主网从未见过872验证图。**

作者v1论文5.1节报告6312训练图/56场景、188测试图/11场景；现用归档的目录计数为6312训练池/59场景、188测试/8场景。图数吻合、场景映射却不同，不能未经逐场景核对就断言具体训练曝光或数据泄漏。[作者原文](https://arxiv.org/html/2503.18055v1#S5.SS1)。目前未找到官方权重逐样本训练清单，这一项列为**尚未完成的来源覆盖核验**。

因此872张可称本次新增模块的保留验证，但不宣称整个系统完全未见；需要独立确认时，应核对预训练覆盖并采用未参与探索的新场景。数据来源问题没有改变已归档的相对T1/候选数值，也不是近期零覆盖的已证实根因。[本次清单核验](../data/data_split_exposure_audit.json)。

'''
if '## 补查：运行时' not in t:t=t.replace('## 文献依据与适用范围',addition+'## 文献依据与适用范围')
report.write_text(t,encoding='utf-8')
story=W/'output/组会汇报主线与讲稿.md';t=story.read_text(encoding='utf-8').replace('建议12页主线','建议13页主线')
if '|7|怎样保护' in t:
 start=t.index('|7|怎样保护');end=t.index('|附录A',start)
 t=t[:start]+'''|7|局部潜力与实际输出差多大|浮点收益、全零动作、GT理想动作同图对照|完整验证候选−0.017dB/最终0/GT参照+0.256dB；参照不可部署|
|8|怎样保护好的区域|从候选到0/.25/.5/1动作的区域保护图|质量门控、局部比较、动作成本、需求引导各在解决不同问题；不用实验年表|
|9|收益判断学会了吗|当前动作收益排序相关性、训练监测|相关接近0；训练32裁剪监测无持续下降，不声明收敛|
|10|为什么最终不改了|训练安全风险覆盖曲线、全零动作图|非零覆盖都不满足预定风险，零覆盖避免损害但研究目标失败|
|11|修复输入信息损失为什么还不够|输入量级与抵消证据|补浮点未证实有效；诊断标准化与正式头输入条件不同|
|12|技术审查发现了什么|三条可核验问题与成本二次结构|输入量级、混合目标、三个独立动作成本；区分事实与根因假设|
|13|下一步怎样得到清晰结论|训练可拟合、场景泛化、整图收益的判断流程|固定候选、一次一个变量、匹配预算；不放松门槛制造提升|
'''+t[end:]
if '补充评估边界：872验证' not in t:t+='\n补充评估边界：872验证是当前新增模块的保留数据，来自官方数据训练目录；官方预训练权重逐样本曝光清单尚未核到。论文与发布数据场景映射不一致，不能称整个系统完全未见验证场景。见技术审查补查段。\n'
story.write_text(t,encoding='utf-8')
with (W/'data/module_inventory.csv').open('w',newline='',encoding='utf-8-sig') as f:
 wr=csv.DictWriter(f,fieldnames=['owner','name','kind','direct_parameters']);wr.writeheader()
 allrows=[('enhancer_system',audit['module_inventory']),('PolarFree_net_g',audit['task_modules'])]+[(k,v['modules']) for k,v in audit['priors'].items()]
 for owner,r in allrows:
  for x in r:wr.writerow(dict(owner=owner,**{k:x[k] for k in ['name','kind','direct_parameters']}))
 # Prior leaf counts must sum to their parent parameter count.
 for k,v in audit['priors'].items():assert sum(x['direct_parameters'] for x in v['modules'])==v['parameters']
# Package the frozen float inputs that support case figures and scale plots.
cases=W/'data/cases';cases.mkdir(exist_ok=True)
for scene,x in e['cases'].items():
 shutil.copy2(Path(x['map_source']),cases/f'{scene}_actual_costs_and_source_probe.npz')
 shutil.copy2(A/f'downstream/metrics/confidence_examples/validation/scene_{scene}.npz',cases/f'{scene}_confidence.npz')
shutil.copy2(A/'diagnostics/feature_cache/safety.npz',W/'data/fixed_safety_features.npz')
code=W/'sources/current_model_code';code.mkdir(exist_ok=True)
for name in ['gain22.py','action_gate.py','enhancer.py','feedback.py','error_guidance.py','bridge.py','study.py','staged.py','acceptance.py','quality_metrics.py']:
 shutil.copy2(A/'scripts'/name,code/name)
shutil.copy2(A/'source/rgb_final_confirmation/polarfree.py',code/'polarfree_wrapper.py')
shutil.copy2(A/'source/rgb_final_confirmation/evaluation.py',code/'official_evaluation.py')
# Static PDF shares the reviewed source PNGs; editable data/diagrams are in PPTX/SVG.
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader
pdf=W/'output/置信与循环纠错_静态预览.pdf';canvas=Canvas(str(pdf),pagesize=(960,540),pageCompression=1)
canvas.setTitle('R2P confidence and recurrent correction review - static preview')
slides=sorted((W/'build/slides').glob('slide-*.png'));assert len(slides)==18
for im in slides:canvas.drawImage(ImageReader(str(im)),0,0,width=960,height=540);canvas.showPage()
canvas.save()
from pypdf import PdfReader
import pypdfium2 as pdfium
import numpy as np
from PIL import Image
assert len(PdfReader(pdf).pages)==18
doc=pdfium.PdfDocument(str(pdf));errors=[];render=W/'build/pdf_review';render.mkdir(exist_ok=True)
for i,page in enumerate(doc):
 im=page.render(scale=4/3).to_pil().convert('RGB');src=np.array(Image.open(slides[i]).convert('RGB'));out=np.array(im);assert out.shape==src.shape
 errors.append(float(np.abs(src.astype(np.int16)-out.astype(np.int16)).mean()))
 if i in [0,6,17]:im.save(render/f'page_{i+1:02d}.png')
 page.close()
assert max(errors)<3,errors
(W/'data/pdf_preview_receipt.json').write_text(json.dumps(dict(pages=18,static_raster_preview=True,editable_counterpart='PPTX; native charts and diagrams',source_slide_render_max_mean_pixel_difference=max(errors),vector_figures='separate SVG/PDF scientific figures'),ensure_ascii=False,indent=2),encoding='utf-8')
readme='''# 交付说明

完整交付目录包含当前网络审查、完整结构图、真实数据集配图，以及18页可编辑组会初稿（13页主线+5页附录，约15–20分钟）。**“网络结构与科研配图.zip”仅打包审查/讲稿、图件、浮点证据和源码；PPTX及静态预览PDF另行独立交付，不在该素材包内。**

- `output/当前网络技术审查.md`：代码定位、已确认事实、设计风险、待验证建议及来源数据边界。
- `output/组会汇报主线与讲稿.md`：问题驱动的页面安排与口头讲稿。
- `output/置信与循环纠错_组会初稿.pptx`：文字、结构节点/箭头和主要数据图表可编辑；复杂热图/密度图为科研栅格图，浮点源另存。
- `output/置信与循环纠错_静态预览.pdf`：18页静态阅览稿；编辑请用PPTX。
- `figures/arch_*.svg`：5张可编辑原尺寸完整结构图，含冻结、历史训练和遗留路径；`figures/specs`保留源JSON。
- `figures/01_*.{png,svg,pdf}` 等：12组科研配图，含样本、置信/误差、候选收益、正负裁剪、风险覆盖、排序、输入尺度、场景差异、GT参照和训练监测。
- `data/module_inventory.csv`：所有登记子模块；重复模块实例逐项列出，登记不等于当前前向使用。
- `data/cases`：固定3个验证场景首帧的原始浮点证据；主讲例图固定场景30。
- `data/fixed_safety_features.npz`：输入量级/区域抵消图使用的已有安全裁剪缓存。GT仅标签，不作为部署输入。
- `sources`：所用原始数据图像和核验过的实际结构源码副本。

图像显示用PNG与浮点评价分开。ROI属于事后解释示例，GT理想动作不可部署。近期全零覆盖仍是方法失败。没有新GPU计算或训练，也没有修改旧实验归档/模型。

复现需要现有第22项主包和bundled Python/Node；绘图用Matplotlib，结构图用现有FigureSpec，PPT用Artifact Tool。当前数据/模型的主权重SHA及逐文件证据在JSON与完整实验报告中。图表为Excel兼容保存10位有效数字，原始研究数据保持原精度。更改数据或模型需要新的实验协议。
'''
(W/'README.md').write_text(readme,encoding='utf-8')
caption='''# 配图索引

|编号|主题|数据与限制|
|---|---|---|
|01|置信图与真实误差|固定验证30首帧，floatNPZ，C不是概率|
|02|T1/候选/GT及实际/理想动作|实际全零；GT图不可部署|
|03|同图正负128×128裁剪|有纹理区域的事后极值，只做解释|
|04|质量与给定修正收益|同图Spearman与固定步长密度图|
|05|训练内风险覆盖|偏差/安全仅训练角色，空接受集不能称100%准确|
|06|最终头各动作收益排序|逐图平均，无独立样本显著性结论|
|07|验证场景候选差异|8场景，PSNR与SSIM分别报告|
|08|区域平均的正负抵消|已有固定安全缓存，量级证据不是根因证明|
|09|实际输出与GT选择参照差距|完整872验证，GT参照不可部署|
|10|训练监测|固定32裁剪偏差角色，不是选择/收敛证明|
|11|50通道输入量级|固定安全采样，ridge标准化/正式头直接拼接|
|12|正负裁剪原图位置|坐标与03一致，蓝/红框只做定位|

结构图arch01到05依次为总图、增强器、PolarFree、Z/Q反馈、H/动作判断。RGB图片仅显示时8位化，所有误差收益分析取浮点成本；真实来源文件、色标与ROI坐标见evidence_summary.json。
'''
(W/'output/配图索引.md').write_text(caption,encoding='utf-8')
items=[]
for folder in ['figures','data','sources','scripts']:
 for file in (W/folder).rglob('*'):
  if file.is_file() and file.name!='实验记录_补充前快照.md':items.append(file)
items += [W/'README.md',report,story,W/'output/配图索引.md']
manifest=[dict(file=x.relative_to(W).as_posix(),bytes=x.stat().st_size,sha256=sha(x)) for x in sorted(items)]
mp=W/'output/素材包清单.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
pack=W/'output/网络结构与科研配图.zip'
with zipfile.ZipFile(pack,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
 for x in sorted(items):z.write(x,x.relative_to(W).as_posix())
 z.write(mp,'素材包清单.json')
with zipfile.ZipFile(pack) as z:
 assert z.testzip() is None
 for x in manifest:assert hashlib.sha256(z.read(x['file'])).hexdigest()==x['sha256']
receipt=dict(review_complete=True,new_training=False,new_gpu=False,current_checkpoint=audit['checkpoint_sha256'],slides=18,main_slides=13,appendix_slides=5,architecture_figures=5,scientific_figure_groups=12,module_inventory_rows=sum(len(r) for _,r in allrows),static_preview_pages=18,plot_library='Matplotlib',structure_library='FigureSpec',presentation_library='Artifact Tool',official_pretraining_exposure_pending=True,files=[dict(file=x.name,bytes=x.stat().st_size,sha256=sha(x)) for x in [report,story,W/'output/置信与循环纠错_组会初稿.pptx',pdf,pack]],material_package_files=len(manifest))
(W/'output/交付核验.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
# Maintain the user-facing history without inventing a new training experiment.
journal=P/'实验记录.md';old=journal.read_text(encoding='utf-8');tag='## 补充记录：2026-10-13技术审查与组会整理（没有新训练）'
if tag not in old:
 backup=W/'sources/实验记录_补充前快照.md';backup.write_text(old,encoding='utf-8')
 old+='\n\n'+tag+'''\n\n这次没有再跑一组模型，而是回头检查已有设计和代码，把置信探索整理成能够讨论的问题。两轮确实执行，第一轮和候选冻结、GT隔离等现成检查仍成立；没有发现可以据此改写第22项结果的执行漏洞。\n\n发现的重点是：简单诊断会标准化输入，而正式判断直接拼接不同量级通道；训练目标同时让输出估计成本和偏向好动作；三个动作成本没有利用它们应满足的二次关系。这些是可核验的设计事实，尚不能说哪一个就是失败根因。\n\n另外872验证来自原数据训练目录，虽然我们自己的训练/验证场景无交叉，仍需核对官方预训练权重是否见过它们；作者论文与当前发布数据的场景计数不同，不能未经核对宣称整个系统完全未见。\n\n已经画出5张完整分层结构图，包含增强器、PolarFree先验/主网、质量/需求头、跨轮状态、残差修正、动作判断、对齐及遗留路径。用真实验证场景30的图片与浮点热图说明“哪里差”和“某次修改是否有效”的差别，并准备18页问题驱动组会初稿。近期的负面结果、GT参照不可部署及缺匹配对照等限制均保留。\n\n[技术审查](analysis/confidence_review_20261013/output/当前网络技术审查.md) · [汇报主线](analysis/confidence_review_20261013/output/组会汇报主线与讲稿.md) · [组会初稿](analysis/confidence_review_20261013/output/置信与循环纠错_组会初稿.pptx)\n'''
 journal.write_text(old,encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False))
