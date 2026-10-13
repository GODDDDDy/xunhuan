import json,sys,torch
from common import ROOT,status
import gain22,staged
torch.set_num_threads(4)
action=sys.argv[1];cfg=json.loads((ROOT/'config'/('gain_preflight.json' if action in ('verify','benchmark') else 'locked_run.json')).read_text())
device=torch.device('cpu' if action=='verify' else 'cuda');status(action)
if action=='verify':gain22.verify(cfg)
elif action=='benchmark':gain22.benchmark(cfg,device)
elif action=='diagnostic':gain22.diagnostic(cfg,device)
elif action=='initialize':gain22.initialize(cfg,device)
elif action=='judgement':staged.train_phase(cfg,device,'judgement')
elif action=='calibration':staged.calibrate(cfg,device)
elif action=='mechanism':staged.mechanism(cfg,device)
elif action in ('validation','test'):staged.evaluate(cfg,device,action)
else:raise ValueError(action)
