from typing import Dict, Any, List
class Selector:
    def __init__(self,cfg:Dict[str,Any]): self.k=cfg.get('selection',{}).get('k',48)
    def __call__(self,items:List[Dict[str,Any]]): return items[:self.k]
