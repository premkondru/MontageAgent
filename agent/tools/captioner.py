from typing import Dict, Any, List
class Captioner:
    def __init__(self,cfg:Dict[str,Any]): self.cfg=cfg
    def __call__(self,items_or_clusters,cluster_mode:bool=False):
        posts=[]
        if not cluster_mode:
            for it in items_or_clusters:
                posts.append({'images':[it['path']],'caption':'Capturing the vibe at Montage! #IITGuwahati #Montage','hashtags':['#IITGuwahati','#Montage','#PhotographyClub'],'labels':it.get('labels',[]),'cluster_id':None})
        else:
            for cl in items_or_clusters:
                paths=[m['path'] for m in cl['items']]
                labels=[]; [labels.append(l) for m in cl['items'] for l in m.get('labels',[]) if l not in labels]
                cid = int(cl.get("cluster_id", 0))
                caption = f"Highlights from the event — set {cid + 1}. #IITGuwahati #Montage"
                posts.append({
                "images": paths,
                "caption": caption,
                "hashtags": ["#IITGuwahati", "#Montage", "#PhotographyClub"],
                "labels": labels,
                "cluster_id": cid,
                })
        return posts
