"""Online PIDSR training, with observation-only inputs and fixed scene splits."""
from collections import OrderedDict
from pathlib import Path
import cv2,numpy as np,torch
from torch.utils.data import Dataset
from vendor.pidsr_data_official import PIDSRData

VAL_SCENES=['011','013','018','020','032','059','078','094','095','097']
ANGLES=(0,45,90,135)

def augment(patch,mode):
    # Transform analyzer labels with spatial coordinates. Generate CPFA afterwards.
    if mode==1:return np.ascontiguousarray(np.flip(patch,axis=1)[...,[0,3,2,1],:])
    if mode==2:return np.ascontiguousarray(np.rot90(patch,k=1,axes=(0,1))[...,[2,3,0,1],:])
    return np.ascontiguousarray(patch)

def degrade(patch):
    # Input is four BGR uint16 angle images stacked as H,W,4,3.
    clean=patch.astype(np.float32)/65535.
    h,w=clean.shape[:2]
    lr=[cv2.resize(clean[:,:,i]*255,(w//2,h//2),interpolation=cv2.INTER_LINEAR) for i in range(4)]
    bayer=PIDSRData.polar_mosaic(None,*lr,COLOR_BAYER_PATTERN='RGGB')
    # This is the exact official OpenCV conversion before canonical RGB reordering.
    inp=[cv2.cvtColor(b,cv2.COLOR_BAYER_RG2RGB).astype(np.float32)/255 for b in bayer]
    target_llr=[cv2.resize(clean[:,:,i],(w//4,h//4),interpolation=cv2.INTER_LINEAR) for i in range(4)]
    def tensor(blocks):
        x=np.concatenate([x[...,::-1] for x in blocks],axis=-1)
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1)))
    return {'polar':tensor(inp),'target':tensor(target_llr)}

class PolarDataset(Dataset):
    def __init__(self,root,split='train',samples=96000,start=0,seed=2026):
        self.root=Path(root);self.split=split;self.samples=samples;self.start=start;self.seed=seed
        all_train=sorted(p.name for p in (self.root/'train').iterdir() if p.is_dir())
        assert len(all_train)==108 and set(VAL_SCENES)<=set(all_train)
        self.scenes=([s for s in all_train if s not in VAL_SCENES] if split=='train' else
                     VAL_SCENES if split=='validation' else
                     sorted(p.name for p in (self.root/'test').iterdir() if p.is_dir()))
        assert len(self.scenes)=={'train':98,'validation':10,'test':30}[split]
        self.cache=OrderedDict()

    def __len__(self):return self.samples if self.split=='train' else len(self.scenes)

    def scene(self,name):
        if name not in self.cache:
            folder=self.root/('test' if self.split=='test' else 'train')/name
            images=[cv2.imread(str(folder/f'RGB_{a}.png'),cv2.IMREAD_UNCHANGED) for a in ANGLES]
            if any(x is None or x.dtype!=np.uint16 or x.shape!=images[0].shape for x in images):
                raise ValueError(f'Expected aligned uint16 PIDSR images: {folder}')
            self.cache[name]=np.stack(images,axis=2)
            while len(self.cache)>2:self.cache.popitem(last=False)
        self.cache.move_to_end(name);return self.cache[name]

    def __getitem__(self,index):
        absolute=self.start+index
        if self.split=='train':
            # Every cycle visits all 98 training scenes, with reproducible permutations.
            # Twelve consecutive crops per scene bound PNG decode overhead;
            # each 1176-sample cycle still visits every training scene.
            cycle,pos=divmod(absolute//12,len(self.scenes))
            order=np.random.default_rng(self.seed+cycle).permutation(len(self.scenes))
            name=self.scenes[int(order[pos])]
            rng=np.random.default_rng(self.seed+absolute*1000003)
            x=self.scene(name);h,w=x.shape[:2];size=512
            y=int(rng.integers((h-size)//4+1))*4;z=int(rng.integers((w-size)//4+1))*4
            patch=augment(x[y:y+size,z:z+size],int(rng.integers(3)))
        else:
            name=self.scenes[index];x=self.scene(name)
            # Official PIDSR evaluation region: y500:1524, x700:1724.
            if x.shape[0]<1524 or x.shape[1]<1724:raise ValueError('Official 1024 ROI exceeds image')
            patch=x[500:1524,700:1724]
        result=degrade(patch);result['scene']=name;return result
