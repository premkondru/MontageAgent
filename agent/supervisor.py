from dataclasses import dataclass
from typing import Dict,Any,List
import traceback
from agent.tools.ingest import Ingestor
from agent.tools.dedupe_quality import DedupeQuality
from agent.tools.categorize import Categorizer
from agent.tools.select_diverse import Selector
from agent.tools.cluster_photos import Clusterer
from agent.tools.captioner import Captioner
from agent.tools.publisher import Publisher

@dataclass
class StepResult:
    name:str
    output:Dict[str,Any]
    success:bool=True
    error:str=""

class Supervisor:
    def __init__(self,cfg:Dict[str,Any]):
        self.cfg=cfg
        self.ingestor=Ingestor(cfg)
        self.dq=DedupeQuality(cfg)
        self.categorizer=Categorizer(cfg)
        self.selector=Selector(cfg)
        self.clusterer=Clusterer(cfg)
        self.captioner=Captioner(cfg)
        self.publisher=Publisher(cfg)
    def run(self)->List[StepResult]:
        S=[]
        try:
            data=self.ingestor(); S.append(StepResult('ingest',{'n_items':len(data)}))
            data=self.dq(data); S.append(StepResult('dedupe_quality',{'n_items':len(data)}))
            data=self.categorizer(data); S.append(StepResult('categorize',{'n_items':len(data)}))
            pick=self.selector(data); S.append(StepResult('select_diverse',{'n_selected':len(pick)}))
            clusters=self.clusterer(pick); 
            metrics = getattr(self.clusterer, 'last_metrics', {})
            S.append(StepResult('cluster', {'n_clusters': len(clusters), **metrics}))
            # NEW: build a per-image label index so the UI can show labels under thumbnails
            label_index = {}
            for cl in clusters:
                for it in cl['items']:
                    label_index[it['path']] = it.get('labels', [])

            # Captioning (posts)
            posts = self.captioner(clusters, cluster_mode=True)

            # Include label_index in the captioner step result
            cap_metrics = getattr(self.captioner, "last_metrics", {})
            S.append(StepResult("captioner", {
                "n_posts": len(posts),
                "posts": posts,
                "label_index": label_index,   # if you already added this earlier
                **cap_metrics                 # ← shows in Streamlit "Pipeline step outputs"
            }))
            if self.cfg.get('publisher',{}).get('enabled',False):
                self.publisher(posts); S.append(StepResult('publisher',{'status':'queued/published'}))
            return S
        except Exception as e:
            S.append(StepResult('error',{},False,str(e)+'\n'+traceback.format_exc())); return S
