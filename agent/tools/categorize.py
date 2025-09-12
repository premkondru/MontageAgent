class Categorizer:
    def __init__(self,cfg): self.cfg=cfg
    def __call__(self,items):
        for it in items: it['labels']=['candid']
        return items
