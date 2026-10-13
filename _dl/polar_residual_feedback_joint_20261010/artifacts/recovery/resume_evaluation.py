"""Resume ONLY predeclared evaluation, keeping the trained graph/config immutable.

The original mechanism entry point imports the task package before adding its
source path. Setting PYTHONPATH corrects execution without editing frozen code.
"""
import hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPECTED_PROTOCOL='87b0691d59aa2ae342e321ff2ecc9534782585453ba16a659cbaeef52614aeeb'
EXPECTED_FINAL='ec7f0e6e79d0dcf21c536dc71958d44514bd7e78a8624bb936f16c02c37e81e8'
sys.path.insert(0,str(ROOT/'scripts'))
from common import json_save,sha


def run():
    cfg=json.loads((ROOT/'config/locked_run.json').read_text())
    budget=json.loads((ROOT/'budget.json').read_text())
    assert cfg['protocol_id']==EXPECTED_PROTOCOL
    assert budget['start_unix']==1791609870.5695198 and budget['deadline_unix']==1791696270.5695198
    assert time.time()<budget['deadline_unix']
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:command=(entry/'cmdline').read_bytes().decode(errors='replace').split('\x00')
        except OSError:continue
        candidates=(ROOT/'scripts/controller.py',ROOT/'scripts/run_stage.py',ROOT/'scripts/watchdog.py',Path(__file__).resolve())
        if any(str(p) in command for p in candidates):raise RuntimeError(f'Study process already exists: {entry.name}')
    assert sha(ROOT/'downstream/checkpoints/final.pt')==EXPECTED_FINAL
    assert all(sha(ROOT/path)==digest for path,digest in cfg['code_sha256'].items())
    selection=json.loads((ROOT/'downstream/selection.json').read_text())
    assert selection['step']==cfg['downstream']['steps']==12000 and selection['sha256']==EXPECTED_FINAL
    import torch
    state=torch.load(ROOT/'downstream/checkpoints/final.pt',map_location='cpu',weights_only=False)
    assert state['step']==12000 and state['protocol_id']==EXPECTED_PROTOCOL
    del state
    preserved=ROOT/'diagnostics/evaluation_recovery_20261010';preserved.mkdir(parents=True,exist_ok=True)
    assert json.loads((ROOT/'status.json').read_text())['stage']=='failed'
    for relative in ('logs/controller.log','logs/mechanism.log','logs/failure.log','status.json',
                     'archive_ready.json','archive_manifest.json','RESULTS_ZH.md','artifacts.tar.gz'):
        source=ROOT/relative
        if source.exists():
            destination=preserved/source.name
            assert not destination.exists(),destination
            shutil.move(str(source),str(destination))
    json_save(preserved/'recovery.json',dict(reason='Mechanism entry-point dependency search path missing before import',
        fix='Set PYTHONPATH to existing, frozen task source directory for evaluation children',
        training_restarted=False,model_or_hyperparameters_changed=False,frozen_code_changed=False,
        checkpoint_step=12000,checkpoint_sha256=EXPECTED_FINAL,protocol_id=EXPECTED_PROTOCOL,
        previous_budget=budget,recorded_unix=time.time()))
    budget['process_group']=os.getpgrp();json_save(ROOT/'budget.json',budget)
    os.environ.update(PYTHONPATH=cfg['paths']['source_root'],CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONDONTWRITEBYTECODE='1',
        TMPDIR=str(ROOT.parent.parent/'tmp_prf'),XDG_CACHE_HOME=str(ROOT/'runtime/cache'),
        TORCH_HOME=str(ROOT.parent/'runtime/cache/torch'))
    with (ROOT/'logs/watchdog.log').open('a') as log:
        wd=subprocess.Popen([sys.executable,str(ROOT/'scripts/watchdog.py')],cwd=ROOT,
                            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    json_save(ROOT/'watchdog.json',dict(pid=wd.pid,process_group=os.getpgrp(),deadline_unix=budget['deadline_unix']))
    json_save(ROOT/'status.json',dict(stage='evaluation_recovered',step=12000,time_unix=time.time()))
    for action in ('mechanism','validation','test'):
        assert time.time()<budget['deadline_unix']
        assert sha(ROOT/'downstream/checkpoints/final.pt')==EXPECTED_FINAL
        with (ROOT/'logs'/f'{action}.log').open('a') as log:
            child_env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
            subprocess.run([sys.executable,str(ROOT/'scripts/run_stage.py'),action],cwd=ROOT,
                           stdout=log,stderr=subprocess.STDOUT,env=child_env,check=True)
    assert sha(ROOT/'downstream/checkpoints/final.pt')==EXPECTED_FINAL
    assert all(sha(ROOT/path)==digest for path,digest in cfg['code_sha256'].items())
    json_save(ROOT/'status.json',dict(stage='completed',step=12000,time_unix=time.time(),evaluation_execution_repaired=True))


if __name__=='__main__':
    try:run()
    except Exception:
        error=traceback.format_exc();(ROOT/'logs/recovery_failure.log').write_text(error)
        json_save(ROOT/'status.json',dict(stage='failed',error=error,time_unix=time.time()));print(error,flush=True)
    finally:
        subprocess.run([sys.executable,str(ROOT/'scripts/archive.py')],cwd=ROOT)
