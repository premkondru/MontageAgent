# agent/tools/cluster_photos.py
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from agent.tools.clip_features import encode_paths, encode_texts
from sklearn.metrics import silhouette_score

_DEFAULT_LABELS = [
    "candid","portrait","group photo","stage","audience","speaker",
    "award","sports","food","night","landscape","architecture","indoors","outdoors"
]

class Clusterer:
    def __init__(self,cfg):
        self.cfg=cfg
        self.maxk=cfg.get('cluster',{}).get('max_images_per_post',10)

        emb=cfg.get('embeddings',{}) if isinstance(cfg,dict) else {}
        self.model=emb.get('model','ViT-B-32')
        self.pretrained=emb.get('pretrained','laion2b_s34b_b79k')
        self.device=emb.get('device','cpu')

        c=cfg.get('cluster',{}) if isinstance(cfg,dict) else {}
        self.use_clip=bool(c.get('use_clip',True))
        self.k_cfg=c.get('k','auto')

        # Fusion knobs
        self.use_label_features=bool(c.get('use_label_features',False))
        self.label_weight=float(c.get('label_weight',0.35))
        self.label_norm=(c.get('label_norm','zscore') or 'none').lower()
        self.fuse_norm=bool(c.get('fuse_normalize',True))

        # Label vocab: prefer categorize.labels, else default
        self.labels=(cfg.get('categorize',{}) or {}).get('labels', _DEFAULT_LABELS)

    def _auto_k(self, n):
        return int(max(1, min(12, round(np.sqrt(max(1, n/2))))))

    def _color_hist(self,p):
        try:
            im=Image.open(p).convert('RGB').resize((64,64))
            arr=np.asarray(im,dtype=np.float32)/255.0
            flat=arr.reshape(-1,3); h,_=np.histogramdd(flat,bins=(8,8,8),range=((0,1),(0,1),(0,1)))
            v=h.ravel().astype('float32'); s=v.sum(); v/=s if s>0 else 1
            return v
        except Exception:
            return np.zeros(512,'float32')

    def _fuse_label_features(self, X_img, paths):
        """
        Build label-score matrix S (cosine sim to label prompts) and fuse:
           X_fused = [ X_img , label_weight * S ]
        Optionally z-score S across samples and L2-normalize the result.
        """
        if not self.labels:
            return X_img
        prompts=[f"a photo of {lab}" for lab in self.labels]
        T = encode_texts(prompts, self.model, self.pretrained, self.device)   # [L, D]
        # If image features are zero (unreadable), similarity will be ~0 — fine.
        S = X_img @ T.T   # [N, L] cosine sims because both sides are unit-norm
        # Normalize label scores if configured
        if self.label_norm == 'zscore' and S.size:
            mu = S.mean(axis=0, keepdims=True)
            sd = S.std(axis=0, keepdims=True) + 1e-6
            S = (S - mu) / sd

        Xf = np.concatenate([X_img, self.label_weight * S], axis=1).astype('float32')

        if self.fuse_norm and Xf.size:
            # L2-normalize each row so scales are comparable
            nrm = np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-8
            Xf = Xf / nrm
        return Xf

    def __call__(self,items):
        if not items: return []
        paths=[it['path'] for it in items]

        # Base feature space
        if self.use_clip:
            X = encode_paths(paths, self.model, self.pretrained, self.device)
        else:
            X = np.stack([self._color_hist(p) for p in paths]).astype('float32')

        # Feature fusion with label scores
        if self.use_label_features and self.use_clip:
            X = self._fuse_label_features(X, paths)

        # Choose k
        if isinstance(self.k_cfg,int) and self.k_cfg>0:
            k=min(self.k_cfg,len(items))
        else:
            k=self._auto_k(len(items))

        if k==1:
            # No silhouette for a single cluster
            self.last_metrics = {"k": 1, "fused": bool(self.use_label_features)}
            return [{'cluster_id':0,'items':items[:self.maxk]}]

        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        y = km.fit_predict(X)

        # --- Quick silhouette score (cosine; fallback to euclidean) ---
        sil = None
        try:
            if len(set(y)) > 1 and len(items) > len(set(y)):
                try:
                    sil = silhouette_score(X, y, metric="cosine")
                except Exception:
                    sil = silhouette_score(X, y)  # euclidean fallback
        except Exception:
            sil = None

        if sil is not None:
            sil = float(round(sil, 3))
            print(f"[cluster] k={k} fused={self.use_label_features} silhouette={sil}")
            self.last_metrics = {"k": k, "fused": bool(self.use_label_features), "silhouette": sil}
        else:
            self.last_metrics = {"k": k, "fused": bool(self.use_label_features), "silhouette": None}

        # Build clusters
        clusters=[]
        for cid in range(k):
            mem=[items[i] for i,lab in enumerate(y) if lab==cid][:self.maxk]
            clusters.append({'cluster_id':cid,'items':mem})
        return clusters

