import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from agent.tools.clip_features import encode_paths

class Clusterer:
    def __init__(self,cfg):
        self.cfg=cfg
        self.maxk=cfg.get('cluster',{}).get('max_images_per_post',10)
        emb=cfg.get('embeddings',{})
        self.model=emb.get('model','ViT-B-32'); self.pretrained=emb.get('pretrained','laion2b_s34b_b79k'); self.device=emb.get('device','cpu')
        self.use_clip=bool(cfg.get('cluster',{}).get('use_clip',True))
    def __call__(self,items):
        if not items: return []
        if self.use_clip:
            X=encode_paths([it['path'] for it in items], self.model, self.pretrained, self.device)
        else:
            def hist(p):
                try:
                    im=Image.open(p).convert('RGB').resize((64,64))
                    arr=np.asarray(im,dtype=np.float32)/255.0
                    flat=arr.reshape(-1,3); h,_=np.histogramdd(flat,bins=(8,8,8),range=((0,1),(0,1),(0,1)))
                    v=h.ravel().astype('float32'); s=v.sum(); v/=s if s>0 else 1
                    return v
                except Exception: return np.zeros(512,'float32')
            X=np.stack([hist(it['path']) for it in items]).astype('float32')
        kcfg=self.cfg.get('cluster',{}).get('k','auto')
        if isinstance(kcfg,int) and kcfg>0: k=min(kcfg,len(items))
        else:
            n=len(items); k=int(max(1,min(12,round(np.sqrt(max(1,n/2))))))
        if k==1: return [{'cluster_id':0,'items':items[:self.maxk]}]
        km=KMeans(n_clusters=k, n_init=10, random_state=42); labels=km.fit_predict(X)
        clusters=[]
        for cid in range(k):
            mem=[items[i] for i,l in enumerate(labels) if l==cid][:self.maxk]
            clusters.append({'cluster_id':cid,'items':mem})
        return clusters
