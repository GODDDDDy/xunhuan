"""Execution-only NumPy scalar serialization; preserve frozen algorithm code."""
import hashlib,json,pathlib,runpy,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'source'))
import common
original=common.json_save
def native(value):
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,dict):return {k:native(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [native(v) for v in value]
 return value
assert json.loads(json.dumps(native({'n':np.int64(3),'v':np.float64(.2)})))=={'n':3,'v':.2}
def json_save(path,value):return original(path,native(value))
common.json_save=json_save
import gain22
if sys.argv[1]=='diagnostic':
 receipt=json.loads((ROOT/'recovery/cache_receipt.json').read_text())
 def cached_records(model,task,legacy,cfg,raw,device,role):
  p=ROOT/'diagnostics/feature_cache'/f'{role}.npz'
  assert hashlib.sha256(p.read_bytes()).hexdigest()==receipt[p.name]
  with np.load(p) as cached:return {k:cached[k].copy() for k in cached.files}
 gain22.crop_records=cached_records
runpy.run_path(str(ROOT/'scripts/run_gain.py'),run_name='__main__')
