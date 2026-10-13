"""Small reproducibility helpers shared by both research training stages."""
import hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def json_save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(path)

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)

def seed(value=2026):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])

def guard_gpu():
    budget=ROOT/'budget.json'
    if budget.exists() and time.time()>=json.loads(budget.read_text())['deadline_unix']:
        raise RuntimeError('GPU authorization deadline has passed')

def status(stage,**extra):
    value=dict(stage=stage,time_unix=time.time(),**extra)
    json_save(ROOT/'status.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)
