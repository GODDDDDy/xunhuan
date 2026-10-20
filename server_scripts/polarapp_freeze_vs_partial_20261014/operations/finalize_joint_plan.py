"""Before GPU/lock, ensure both new arms each sample all5440 training images."""
import json,sys,torch
from pathlib import Path
from PIL import Image
root=Path(sys.argv[1])
sys.path.insert(0,str(root/'scripts'))
from runtime import atomic_json,sha
assert not (root/'budget.json').exists() and not (root/'config/locked_run.json').exists()
path=root/'config/joint_schedule.json'
schedule=json.loads(path.read_text())
draft=json.loads((root/'config/draft_run.json').read_text())
config=json.loads((root/'config/prepare.json').read_text())
from csv import DictReader
with open(config['manifest']) as stream:rows=[r for r in DictReader(stream) if r['split']=='train']
assert len(rows)==5440 and [r['sample_id'] for r in rows]==schedule['sample_ids']
atomic_json(root/'diagnostics/prelock_joint_plan_first_draft.json',schedule)
gen=torch.Generator().manual_seed(202625)
order=torch.randperm(5440,generator=gen).tolist()
order+=torch.randperm(5440,generator=gen).tolist()[:560]
items=[]
for index in order:
    with Image.open(rows[index]['polar_000']) as image:width,height=image.size
    items.append({'sample_index':index,'top':int(torch.randint(0,height-64+1,(1,),generator=gen)),
                  'left':int(torch.randint(0,width-64+1,(1,),generator=gen))})
schedule['items']=schedule['items'][:12000]+items
schedule['additional_plan_seed']=202625
schedule['additional_plan_scope']='new matched full training permutation; not parent unused tail'
assert len(schedule['items'])==18000 and len({i['sample_index'] for i in items})==5440
atomic_json(path,schedule)
draft['schedule_hashes']['joint']=sha(path)
draft['new_joint_window_unique_images']=5440
draft['new_joint_window_plan']='shared fixed full permutation plus560 draws; first12000 items preserved; not unused parent tail'
atomic_json(root/'config/draft_run.json',draft)
atomic_json(root/'diagnostics/prelock_joint_plan_refinement.json',{'CPU_only':True,'before_first_GPU_and_lock':True,
    'reason':'each added3000-step arm must cover all5440 training images','new_joint_unique_images':5440,
    'first12000_source_items_preserved':True,'same_plan_for_both':True,'schedule_sha256':sha(path)})
print('Both added training windows cover all5440 images; original prefix unchanged.')
