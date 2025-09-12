from typing import Dict, Any, List

class Categorizer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.labels = cfg.get("categorize", {}).get("labels", [])

    def __call__(self, items: List[Dict[str, Any]]):
        # Placeholder: tag everything as 'candid'
        for it in items:
            it["labels"] = ["candid"]
        return items
