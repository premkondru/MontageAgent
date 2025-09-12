from typing import Dict, Any, List
class Categorizer:
    def __init__(self,cfg:Dict[str,Any]): self.cfg=cfg
    def __call__(self,items:List[Dict[str,Any]]):
        for it in items: it['labels']=['candid']
        return items
