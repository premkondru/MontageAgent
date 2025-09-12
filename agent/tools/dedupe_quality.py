from typing import Dict, Any, List
from PIL import Image
import os, numpy as np
from agent.tools.clip_features import encode_paths
def _dhash(image_path: str, hash_size: int = 8) -> int:
    try:
        with Image.open(image_path) as img:
            img=img.convert('L').resize((hash_size+1,hash_size), Image.LANCZOS)
            diff=0; b=0; px=list(img.getdata())
            for r in range(hash_size):
                s=r*(hash_size+1)
                for c in range(hash_size):
                    if px[s+c] > px[s+c+1]: diff |= (1<<b)
                    b+=1
            return diff
    except Exception: return 0
def _hamming(a:int,b:int)->int: return bin(a^b).count('1')
class DedupeQuality:
    def __init__(self,cfg:Dict[str,Any]):
        self.cfg=cfg
        d=cfg.get('dedupe',{}) or {}
        self.method=(d.get('method') or 'clip').lower()
        self.clip_threshold=float(d.get('clip_threshold',0.985))
        self.dhash_thresh=int(d.get('near_dup_threshold',5))
        e=cfg.get('embeddings',{}) or {}
        self.model_name=e.get('model','ViT-B-32'); self.pretrained=e.get('pretrained','laion2b_s34b_b79k'); self.device=e.get('device','cpu')
    def __call__(self, items: List[Dict[str,Any]]):
        if not items: return items
        if self.method=='clip':
            paths=[it.get('path') for it in items]
            feats=encode_paths(paths, self.model_name, self.pretrained, self.device)
            keep=[]; reps=[]
            for it,f in zip(items,feats):
                if f.sum()==0:
                    h=_dhash(it['path'])
                    if any(_hamming(h,_dhash(k['path']))<=self.dhash_thresh for k in keep): continue
                    it['clip']=None; keep.append(it)
                else:
                    f=f.astype('float32')
                    if any(float((f*rf).sum())>=self.clip_threshold for rf in reps): continue
                    it['clip']=f; keep.append(it); reps.append(f)
            return keep
        else:
            seen=[]; out=[]
            for it in items:
                p=it.get('path'); 
                if not p or not os.path.exists(p): continue
                h=_dhash(p)
                if any(_hamming(h,sh)<=self.dhash_thresh for sh in seen): continue
                seen.append(h); out.append(it)
            return out
