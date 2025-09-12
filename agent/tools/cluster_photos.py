from typing import Dict, Any, List
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from agent.tools.clip_features import encode_paths
def _hist(img_path:str):
    try:
        im=Image.open(img_path).convert('RGB').resize((64,64))
        import numpy as np
        arr=np.asarray(im,dtype='float32')/255.0
        flat=arr.reshape(-1,3)
        hist,_=np.histogramdd(flat,bins=(8,8,8),range=((0,1),(0,1),(0,1)))
        feat=hist.astype('float32').ravel(); s=feat.sum(); 
        if s>0: feat/=s
        return feat
    except Exception: 
        return np.zeros(8*8*8,dtype='float32')
class Clusterer:
    def __init__(self,cfg:Dict[str,Any]):
        self.cfg=cfg; self.max_images_per_post=cfg.get('cluster',{}).get('max_images_per_post',10)
        e=cfg.get('embeddings',{}) or {}; self.model_name=e.get('model','ViT-B-32'); self.pretrained=e.get('pretrained','laion2b_s34b_b79k'); self.device=e.get('device','cpu')
        self.use_clip=bool(cfg.get('cluster',{}).get('use_clip',True))
    def __call__(self,items:List[Dict[str,Any]]):
        if not items: return []
        if self.use_clip:
            feats=[]; need_paths=[]; need_idx=[]
            for i,it in enumerate(items):
                v=it.get('clip'); 
                if isinstance(v,np.ndarray) and v.size>0: feats.append(v)
                else: feats.append(None); need_paths.append(it['path']); need_idx.append(i)
            if need_paths:
                new=encode_paths(need_paths, self.model_name, self.pretrained, self.device)
                for j,i in enumerate(need_idx): items[i]['clip']=new[j]; feats[i]=new[j]
            X=np.stack(feats,0).astype('float32')
        else:
            X=np.stack([_hist(it['path']) for it in items],0).astype('float32')
        k_cfg=self.cfg.get('cluster',{}).get('k','auto')
        if isinstance(k_cfg,int) and k_cfg>0: k=min(k_cfg,len(items))
        else:
            n=len(items); 
            k=int(max(1, min(12, round((max(1,n/2))**0.5))))
        if k>len(items): k=len(items)
        if k==1: return [{'cluster_id':0,'items':items[:self.max_images_per_post]}]
        km=KMeans(n_clusters=k, n_init=10, random_state=42)
        labels=km.fit_predict(X)
        clusters=[]
        for cid in range(k):
            members=[items[i] for i,l in enumerate(labels) if l==cid][:self.max_images_per_post]
            clusters.append({'cluster_id':cid,'items':members})
        return clusters
