class Publisher:
    def __init__(self,cfg): self.cfg=cfg
    def __call__(self,posts):
        if not self.cfg.get('publisher',{}).get('enabled',False):
            for p in posts: print('[DRY-RUN] Would publish', len(p['images']), 'images')
