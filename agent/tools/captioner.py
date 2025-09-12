class Captioner:
    def __init__(self,cfg): self.cfg=cfg
    def __call__(self,clusters, cluster_mode=False):
        posts=[]
        for cl in clusters:
            paths=[m['path'] for m in cl['items']]
            cid=int(cl.get('cluster_id',0))
            caption=f"Highlights from the event — set {cid+1}"
            hashtags=['#IITGuwahati','#Montage','#PhotographyClub']
            labels=[]
            for m in cl['items']:
                for lab in m.get('labels',[]):
                    if lab not in labels: labels.append(lab)
            posts.append({'images':paths,'caption':caption,'hashtags':hashtags,'labels':labels,'cluster_id':cid})
        return posts
