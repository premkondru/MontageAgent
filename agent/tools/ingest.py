import os

def _is_img(fn): return fn.lower().endswith(('.jpg','.jpeg','.png'))
class Ingestor:
    def __init__(self,cfg): self.cfg=cfg
    def __call__(self):
        roots=self.cfg.get('ingest',{}).get('dirs',['data/events']); items=[]
        for root in roots:
            if not os.path.isdir(root): continue
            for e in sorted(os.listdir(root)):
                p=os.path.join(root,e)
                if os.path.isdir(p):
                    for f in os.listdir(p):
                        if _is_img(f): items.append({'path':os.path.join(p,f),'day':e,'meta':{}})
                elif _is_img(e):
                    items.append({'path':p,'day':os.path.basename(root),'meta':{}})
        return items
