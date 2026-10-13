import json,sys,torch
from common import ROOT,status
torch.set_num_threads(4)
cfg=json.loads((ROOT/'config/locked_run.json').read_text());device=torch.device('cuda')
action=sys.argv[1]
status(action)
if action=='pretraining':
    import pretrain;pretrain.train(cfg,device)
elif action=='initialize':
    import study;study.initialize(cfg,device)
elif action=='training':
    import study;study.train(cfg,device)
elif action in ('validation','test'):
    import study;study.base.add_source_path(cfg);study.evaluate(cfg,device,action)
else:raise ValueError(action)
