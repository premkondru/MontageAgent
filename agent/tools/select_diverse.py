from typing import Dict, Any, List

class Selector:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.k = cfg.get("selection", {}).get("k", 12)

    def __call__(self, items: List[Dict[str, Any]]):
        # Placeholder: first k
        return items[: self.k]
