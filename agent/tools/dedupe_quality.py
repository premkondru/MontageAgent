import os, numpy as np
from PIL import Image
from agent.tools.clip_features import encode_paths

def _dhash(path,hash_size=8):
    try:
        img=Image.open(path).convert('L').resize((hash_size+1,hash_size), Image.LANCZOS)
        diff=0; bit=0; px=list(img.getdata())
        for r in range(hash_size):
            rs=r*(hash_size+1)
            for c in range(hash_size):
                if px[rs+c]>px[rs+c+1]: diff|=(1<<bit)
                bit+=1
        return diff
    except Exception: return 0

def _ham(a,b): return bin(a^b).count('1')

class DedupeQuality:
    def __init__(self,cfg):
        self.cfg=cfg
        d=cfg.get('dedupe',{})
        self.method=(d.get('method') or 'clip').lower()
        self.clip_th=float(d.get('clip_threshold',0.985))
        self.dh_th=int(d.get('near_dup_threshold',5))
        emb=cfg.get('embeddings',{})
        self.model=emb.get('model','ViT-B-32'); self.pretrained=emb.get('pretrained','laion2b_s34b_b79k'); self.device=emb.get('device','cpu')
    def __call__(self, items):
        if not items: return items
        if self.method=='clip':
            paths=[it['path'] for it in items]
            feats=encode_paths(paths, self.model, self.pretrained, self.device)
            keep=[]; reps=[]
            for it,f in zip(items,feats):
                if f.sum()==0:
                    h=_dhash(it['path'])
                    if any(_ham(h,_dhash(k['path']))<=self.dh_th for k in keep):
                        continue
                    it['clip']=None; keep.append(it)
                else:
                    f=f.astype('float32'); dup=any(float((f*rf).sum())>=self.clip_th for rf in reps)
                    if dup: continue
                    it['clip']=f; keep.append(it); reps.append(f)
            return keep
        else:
            seen=[]; out=[]
            for it in items:
                p=it['path']; h=_dhash(p)
                if any(_ham(h,sh)<=self.dh_th for sh in seen): continue
                seen.append(h); out.append(it)
            return out
