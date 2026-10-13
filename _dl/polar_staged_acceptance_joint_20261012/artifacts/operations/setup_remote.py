import json,pathlib,shutil,hashlib
parent=pathlib.Path('/media/li/data3/25thstudent/DY/929work/pidsr_k2_929')
old=parent/'polar_local_acceptance_joint_20261011'
root=parent/'polar_staged_acceptance_joint_20261012'
assert not root.exists(), 'Refuse to overwrite an existing run'
root.mkdir()
for name in ('scripts','source','config'):
    shutil.copytree(old/name,root/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','code_sha256.json'))
for name in ('logs','diagnostics','operations','recovery','downstream/checkpoints','downstream/logs','downstream/config','downstream/metrics','runtime/cache','pretraining/checkpoints','weights/source_polarfree'):
    (root/name).mkdir(parents=True,exist_ok=True)
for name in ('pretraining/checkpoints/selected.pt',):
    shutil.copy2(old/name,root/name)
for name in ('net_g_latest.pth','net_d_latest.pth','net_le_dm_latest.pth'):
    shutil.copy2(old/'weights/source_polarfree'/name,root/'weights/source_polarfree'/name)
shutil.copy2(old/'downstream/checkpoints/final.pt',root/'weights/reused_final.pt')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert digest(root/'weights/reused_final.pt')=='e8a635e052ec86f9abddeaa6731a3cb85533420aa78d3b0ff52fbac5bd38e299'
cfg=json.loads((old/'config/locked_run.json').read_text().replace(str(old),str(root)))
for key in ('code_sha256','protocol_id','budget'):cfg.pop(key,None)
cfg['reuse'].update(checkpoint=str(root/'weights/reused_final.pt'),sha256=digest(root/'weights/reused_final.pt'),source_run=str(old),source_step=6000,source_protocol_id='70b0f6efd6340c9fea9b286d9b9a311fd639bb0e22c7b539d66edfce274425ba',new_initialization_only=['action_cost_head'],first_500_steps='superseded by staged protocol',later='candidate first, frozen-candidate judgement second')
cfg['downstream'].pop('deadline_unix',None)
cfg['experiment']='polar_staged_acceptance_joint_20261012'
cfg['diagnostic']=dict(frames=28,split='fixed validation proxy',actions=[0.,.25,.5,1.],cell_size=4,oracle='GT RGB MSE argmin among actual action outputs; diagnosis only',metric='existing raw prediction PSNR/SSIM; no new clamp',weight_selection=False,test_used=False)
(root/'config/diagnostic.json').write_text(json.dumps(cfg,indent=2))
for p in (root/'config').glob('*.yaml'):
    p.write_text(p.read_text().replace(str(old),str(root)))
# The new source includes its inherited judgement tensors. Strict transfer must
# accept no missing tensors instead of assuming that the old head is absent.
p=root/'scripts/study.py';s=p.read_text()
s=s.replace("assert set(transfer.missing_keys)=={n for n in model.state_dict() if n.startswith('update_gate.')}","assert not transfer.missing_keys, 'All inherited model tensors must migrate strictly'")
s=s.replace("for n,v in model.state_dict().items() if not n.startswith('update_gate.'))","for n,v in model.state_dict().items())")
p.write_text(s)
shutil.copy2(old/'downstream/metrics/system_step_0.json',root/'diagnostics/source_fixed_t1_reference.json')
print(json.dumps(dict(root=str(root),source_sha256=cfg['reuse']['sha256'])))
