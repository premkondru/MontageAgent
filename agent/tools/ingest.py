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
#sup_pathold = "data" / "events"

def get_writable_base_dir(preferred: str = None) -> Path:
    """
    Return a directory we can write to:
      1) APP_DATA_DIR (if set)
      2) /data          (HF Spaces persistent)
      3) /tmp           (ephemeral)
    """
    candidates = [preferred or os.getenv("APP_DATA_DIR"), "/data", tempfile.gettempdir()]
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        try:
            p.mkdir(parents=True, exist_ok=True)
            test = p / ".rwtest"
            with open(test, "w") as f:
                f.write("ok")
            test.unlink(missing_ok=True)
            return p
        except Exception:
            continue
    raise RuntimeError("No writable directory available")

BASE_DATA_DIR = get_writable_base_dir()
sup_path2 = BASE_DATA_DIR / "data" / "events"

def _is_img(fn): return fn.lower().endswith(('.jpg','.jpeg','.png'))
class Ingestor:
    def __init__(self,cfg): self.cfg=cfg
    def __call__(self):
        roots=self.cfg.get('ingest',{}).get('dirs',[sup_path, sup_path2]); items=[]
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
