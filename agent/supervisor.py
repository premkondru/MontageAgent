from dataclasses import dataclass
import traceback
from typing import Dict, Any, List
from agent.tools.ingest import Ingestor
from agent.tools.dedupe_quality import DedupeQuality
from agent.tools.categorize import Categorizer
from agent.tools.select_diverse import Selector
from agent.tools.cluster_photos import Clusterer
from agent.tools.captioner import Captioner
from agent.tools.publisher import Publisher
@dataclass
class StepResult:
    name: str
    output: Dict[str, Any]
    success: bool = True
    error: str = ""
class Supervisor:
    def __init__(self, config: Dict[str, Any]):
        self.cfg=config
        self.ingestor=Ingestor(self.cfg)
        self.dq=DedupeQuality(self.cfg)
        self.categorizer=Categorizer(self.cfg)
        self.selector=Selector(self.cfg)
        self.clusterer=Clusterer(self.cfg)
        self.captioner=Captioner(self.cfg)
        self.publisher=Publisher(self.cfg)
    def run(self)->List[StepResult]:
        steps=[]
        try:
            data=self.ingestor(); steps.append(StepResult('ingest',{'n_items':len(data)}))
            data=self.dq(data); steps.append(StepResult('dedupe_quality',{'n_items':len(data)}))
            data=self.categorizer(data); steps.append(StepResult('categorize',{'n_items':len(data)}))
            pick=self.selector(data); steps.append(StepResult('select_diverse',{'n_selected':len(pick)}))
            clusters=self.clusterer(pick); steps.append(StepResult('cluster',{'n_clusters':len(clusters)}))
            posts=self.captioner(clusters, cluster_mode=True); steps.append(StepResult('captioner',{'n_posts':len(posts),'posts':posts}))
            if self.cfg.get('publisher',{}).get('enabled',False): self.publisher(posts); steps.append(StepResult('publisher',{'status':'queued/published'}))
            return steps
        except Exception as e:
            steps.append(StepResult('error',{},success=False,error=f"{e}\n{traceback.format_exc()}")); return steps
