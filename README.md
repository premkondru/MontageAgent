MontageAgent
Automates sorting, de-duplication, quality filtering, categorization, captioning, hashtagging, and Instagram publishing for Montage’s event photos. Uses a LoRA-finetuned BLIP-2 captioner, CLIP-based selection, and a multi-agent (Planner–Executor with Supervisor/Critic) orchestration.

Author: Prem Kondru (IIT Guwahati, B.Tech Engineering Physics)


## Project Deliverables

1. **[GitHub Repo – MontageAgent](https://github.com/premkondru/MontageAgent)**
   Source code for the end-to-end project, configs, training scripts, and deliverables.

2. **[Hugging Face Space – montage-agent](https://huggingface.co/spaces/premkondru/montage-agent)**
   Live Streamlit app to demo the workflow: upload photos, run dedupe/clustering, generate captions (Template/BLIP-2), and preview posts.

3. **[AI Agent Architecture (PDF)](https://github.com/premkondru/MontageAgent/blob/main/deliverables/MontageAgent-AIAgentArchitecture.pdf)**
   High-level system design of the Montage Agent, including components, data flow, and integration points.

4. **[Data Science Report (PDF)](https://github.com/premkondru/MontageAgent/blob/main/deliverables/MontageAgent-Data%20Science%20Report.pdf)**
   Formal write-up of datasets, fine-tuning setup, evaluation (BLEU/ROUGE/CLIPScore, human ratings), and key results.

5. **[Code Companion – ChatGPT (PDF)](https://github.com/premkondru/MontageAgent/blob/main/deliverables/MontageAgent-Code-ChatGPT.pdf)**
   Code-focused explanations and snippets generated with ChatGPT to aid understanding and maintenance.

6. **[Architecture Doc – ChatGPT (PDF)](https://github.com/premkondru/MontageAgent/blob/main/deliverables/MontageAgent-ArchitectureDoc-ChatGPT.pdf)**
   Narrative architecture overview produced with ChatGPT

7. **[Data Science Report – ChatGPT (PDF)](https://github.com/premkondru/MontageAgent/blob/main/deliverables/MontageAgent-DataScienceReport-ChatGPT.pdf)**
   Expanded, executive-style DS report drafted via ChatGPT, mirroring experiments and insights

8. **[Product Screenshots (PDF)](https://github.com/premkondru/MontageAgent/blob/main/deliverables/MontageAgent-Screenshots.pdf)**
   Slide-style gallery of key UI screens with placeholders and brief context for each step.

9. **[Screen Recording (MOV)](https://github.com/premkondru/MontageAgent/blob/main/deliverables/MontageAgent-ScreenRecording.mov)**
   Short demo video showcasing the app flow—from image upload to caption generation and post preview.



### TGetting Started

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

pip install -e .

streamlit run ui/streamlit_app.py

**Run the agent pipeline**

```bash
python -m agent.supervisor --config configs/agent.yaml
```

**Train BLIP-2 LoRA**

```bash
accelerate launch training/train_lora_blip2.py --config configs/lora_blip2.yaml
```

**Try the UI**

```bash
streamlit run ui/streamlit_app.py
```

---

Here’s what each part of your repo does, from top to bottom:

---

## Top-level

* **`pyproject.toml` / `setup.py`** – packaging + install metadata so you can `pip install -e .`.
* **`requirements.txt`** – Python deps for runtime/dev.
* **`README.md`** – project overview and quickstart.
* **`.gitignore`** – files/folders Git should skip.

---

## Config & Data

* **`configs/`**

  * **`agent.yaml`** – end-to-end pipeline switches and defaults (e.g., `pipeline: [ingest, dedupe_quality, categorize, cluster, captioner, publisher]`, model/backends, thresholds).
  * **`lora_blip2.yaml`** – training config for BLIP-2 LoRA (paths, lr, batch sizes, scheduler, eval options).
* **`data/`** – raw/processed assets you run the pipeline or training on (e.g., images, JSONL caption files).

---

## Runtime Agent

* **`agent/`** – Python package for the production agent.

  * **`__init__.py`** – package init.
  * **`supervisor.py`** – orchestrator: reads `configs/agent.yaml`, runs stages in order, handles logging/IO.
  * **`tools/`** – each pipeline stage as a tool (single-responsibility modules):

    * **`ingest.py`** – discovers/loads images and metadata into a working manifest.
    * **`dedupe_quality.py`** – near-duplicate removal (CLIP cosine + NMS) and basic quality gates.
    * **`categorize.py`** – applies label classifier/tagger (e.g., “people”, “night”, “food”).
    * **`clip_features.py`** – computes/loads CLIP embeddings used by dedupe/cluster/CLIPScore.
    * **`cluster_photos.py`** – groups images (k/auto, silhouette reporting, optional label fusion).
    * **`captioner.py`** – caption generation:

      * **Template mode** – deterministic templates.
      * **BLIP-2 mode** – LoRA-tuned BLIP-2; includes decoding + optional CLIPScore evaluation.
    * **`publisher.py`** – assembles final post objects (caption + controlled hashtags), dry-run or push.

---

## Checkpoints

* **`checkpoints/lora_blip2_montage/`**

  * **`best/`** – best-metric LoRA adapters (e.g., best ROUGE-L).
  * **`last/`** – most recent training checkpoint.
  * Used by `captioner.py` when BLIP-2 mode is selected.

---

## Training

* **`training/`** – scripts/notebooks for model fine-tuning and evaluation (e.g., `train_lora_blip2.py`, metric code, utilities).

  * Reads `configs/lora_blip2.yaml`, saves to `checkpoints/lora_blip2_montage/`.

---

## Deliverables

* **`deliverables/`** – generated artifacts you share (report PDFs, slide exports, tables, sample posts, charts).

---

## Publisher

* **`publisher/`** – (optional) targets/integrations (e.g., export to CSV/JSON, API wrappers for IG scheduler). Keeps integration code separate from core agent.

---

## UI

* **`ui/streamlit_app.py`** – lightweight front end to run the pipeline interactively:

  * Upload/select images
  * Toggle Template vs BLIP-2
  * Preview captions/hashtags
  * Inspect CLIPScore/silhouette and download outputs

---



### Mental model

* **`agent/`** = production pipeline (stage-by-stage tools).
* **`training/` + `checkpoints/`** = model improvement loop.
* **`configs/`** = everything is driven by YAML.
* **`ui/`** = demo/run it without code.
* **`deliverables/`** = what you present/share.

This layout cleanly separates *orchestration*, *ML training*, and *presentation*, making it easy to iterate on each without breaking the others.

