from typing import Dict, Any, List

class Captioner:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def __call__(self, items: List[Dict[str, Any]]):
        posts = []
        for it in items:
            caption = "Capturing the vibe at Montage! #IITGuwahati #Montage"
            hashtags = ["#IITGuwahati", "#Montage", "#PhotographyClub"]
            posts.append({
                "image_path": it["path"],
                "caption": caption,
                "hashtags": hashtags,
                "labels": it.get("labels", [])
            })
        return posts
