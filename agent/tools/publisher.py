from typing import Dict, Any, List

class Publisher:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def __call__(self, posts: List[Dict[str, Any]]):
        dry = self.cfg.get("publisher", {}).get("dry_run", True)
        for p in posts:
            if dry:
                print(f"[DRY-RUN] Would publish: {p['image_path']} | {p['caption']}")
            else:
                # TODO: call Instagram Graph API here
                pass
