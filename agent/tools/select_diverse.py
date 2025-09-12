class Selector:
    def __init__(self,cfg): self.k=cfg.get('selection',{}).get('k',48)
    def __call__(self,items): return items[:self.k]
