import os
import sys, os, time, yaml, json, importlib.util, re
from pathlib import Path
from io import BytesIO
import tempfile
# Ensure repo root is importable even if Streamlit launched from elsewhere
repo_root = Path(__file__).resolve().parents[1].parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
sup_path = repo_root / "data" / "events"

def _is_img(fn): return fn.lower().endswith(('.jpg','.jpeg','.png'))
class Ingestor:
    def __init__(self,cfg): self.cfg=cfg
    def __call__(self):
        roots=self.cfg.get('ingest',{}).get('dirs',[sup_path]); items=[]
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
