import json,sys,torch
from common import ROOT,status
torch.set_num_threads(4)
action=sys.argv[1]
config_name='preflight.json' if action=='benchmark' else 'locked_run.json'
cfg=json.loads((ROOT/'config'/config_name).read_text());device=torch.device('cuda')
status(action)
if action=='benchmark':
    import study;study.benchmark(cfg,device)
elif action=='initialize':
    import study;study.initialize(cfg,device)
elif action=='training':
    import study;study.train(cfg,device)
elif action in ('validation','test'):
    import study;study.base.add_source_path(cfg);study.evaluate(cfg,device,action)
else:raise ValueError(action)
