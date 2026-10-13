import json,sys,torch
from common import ROOT,status
import staged,short_verify,short_diagnostic
torch.set_num_threads(4)
action=sys.argv[1];cfg=json.loads((ROOT/'config'/('short_preflight.json' if action in ('verify','benchmark') else 'locked_run.json')).read_text())
device=torch.device('cpu' if action=='verify' else 'cuda');status(action)
if action=='verify':short_verify.verify(cfg)
elif action=='benchmark':staged.benchmark(cfg,device)
elif action=='source_diagnostic':short_diagnostic.run(cfg,device)
elif action=='initialize':staged.initialize(cfg,device)
elif action in ('candidate','judgement'):staged.train_phase(cfg,device,action)
elif action=='calibration':staged.calibrate(cfg,device)
elif action=='mechanism':staged.mechanism(cfg,device)
elif action in ('validation','test'):staged.evaluate(cfg,device,action)
else:raise ValueError(action)
