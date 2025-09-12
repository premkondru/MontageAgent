class Publisher:
    def __init__(self,cfg): self.cfg=cfg
    def __call__(self,posts):
        dry=self.cfg.get('publisher',{}).get('dry_run',True)
        for p in posts:
            if dry:
                print(f"[DRY-RUN] Would publish {len(p['images'])} images | {p['caption']}")
            else:
                pass
