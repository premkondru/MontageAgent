import os
from typing import Dict, Any, List
class Ingestor:
    def __init__(self, cfg: Dict[str, Any]): self.cfg=cfg
    def __call__(self)->List[Dict[str,Any]]:
        roots=self.cfg.get('ingest',{}).get('dirs',['data/events'])
        items=[]
        for root in roots:
            if not os.path.isdir(root): continue
            for entry in sorted(os.listdir(root)):
                path=os.path.join(root,entry)
                if os.path.isdir(path):
                    for fname in os.listdir(path):
                        if fname.lower().endswith(('.jpg','.jpeg','.png')):
                            items.append({'path': os.path.join(path,fname), 'day': entry, 'meta': {}})
                elif entry.lower().endswith(('.jpg','.jpeg','.png')):
                    items.append({'path': path, 'day': os.path.basename(root), 'meta': {}})
        return items
