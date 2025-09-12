# Montage Photo Agent

**Author:** Prem Kondru  
**University:** Indian Institute of Technology (IIT) Guwahati  
**Department:** B.Tech, Engineering Physics  

Automates sorting, de-duplication, quality filtering, categorization, captioning, hashtagging, and optional Instagram publishing for Montage’s event photos. 
Uses a LoRA-finetuned BLIP-2 captioner, CLIP-based selection, and a multi-agent (Planner–Executor with Supervisor/Critic) orchestration. 

## Quickstart

```bash
# 1) Create & activate venv (example for Python 3.10+)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install
pip install -r requirements.txt

# 3) Run Streamlit UI
streamlit run ui/streamlit_app.py

# 4) Train LoRA (placeholder script)
python models/train_lora_blip2.py --config configs/lora_blip2.yaml
```

## Repo Structure

```
montage-photo-agent/
├─ agent/
│  ├─ supervisor.py
│  ├─ tools/
│  │   ├─ ingest.py
│  │   ├─ dedupe_quality.py
│  │   ├─ categorize.py
│  │   ├─ select_diverse.py
│  │   ├─ captioner.py
│  │   └─ publisher.py
│  └─ mcp/
│      ├─ fs_server/README.md
│      ├─ style_rag_server/README.md
│      └─ instagram_server/README.md
├─ models/
│  ├─ train_lora_blip2.py
│  └─ checkpoints/  (gitignored)
├─ data/
│  ├─ events/
│  │   ├─ sample_event_day1/
│  │   │   └─ (place images here)
│  │   └─ sample_event_day2/
│  └─ style/
│      ├─ hashtag_bank.csv
│      └─ past_captions.jsonl
├─ eval/
│  ├─ metrics.py
│  ├─ ablations.py
│  └─ human_eval_protocol.md
├─ ui/
│  └─ streamlit_app.py
├─ docs/
│  ├─ architecture.md
│  ├─ data_science_report.md
│  └─ agent_patterns_mapping.md
├─ configs/
│  ├─ agent.yaml
│  └─ lora_blip2.yaml
├─ .gitignore
└─ requirements.txt
```

## Design Patterns Referenced
- Anthropic’s “Building effective agents” principles (tool-centric, small reliable steps).
- Agent Design Pattern Catalogue (arXiv:2405.10467).
- Planner→Executor with Critic/Reflexion loop; Supervisor–Workers for specialization.

## Notes
- Replace sample data with your own. 
- Instagram publishing requires a Professional account linked to a Page and the Instagram Graph API.
- The training script is a template; wire in your environment (accelerator, dataset paths).
