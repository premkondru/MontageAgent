import os
from typing import Dict, Any, List

class Ingestor:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def __call__(self) -> List[Dict[str, Any]]:
        root = "data/events"
        items = []
        for day in sorted(os.listdir(root)):
            day_path = os.path.join(root, day)
            if not os.path.isdir(day_path):
                continue
            for fname in os.listdir(day_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    items.append({
                        "path": os.path.join(day_path, fname),
                        "day": day,
                        "meta": {}
                    })
        return items
