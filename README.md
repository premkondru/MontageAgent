# MontageAgent

Automates sorting, de-duplication, quality filtering, categorization, captioning, hashtagging, and Instagram publishing for **Montage’s event photos**.
Uses a **LoRA-finetuned BLIP-2 captioner**, **CLIP-based selection**, and a **multi-agent (Planner–Executor with Supervisor/Critic) orchestration**.

**Author:** Prem Kondru (IIT Guwahati, B.Tech Engineering Physics)

---

## Overview

MontageAgent is an **AI-powered agent** developed for IIT Guwahati’s Photography Club (*Montage*) to streamline event photo management and Instagram publishing.

Currently, every event produces hundreds of images that require time-intensive manual effort:

* removing duplicates,
* sorting and choosing the best shots,
* categorizing by theme,
* writing captions in a consistent tone,
* packaging into Instagram carousels.

**MontageAgent automates this workflow end-to-end:**
**Ingest → Dedupe → Cluster → Caption → Export**

Human oversight remains built-in via a **Streamlit GUI** that allows selective edits and adjustments before final publishing.

---

## ⚡ GUI Quickstart (No Setup Needed)

You can try MontageAgent right away via its deployed **Hugging Face Space**:

👉 [**montage-agent on Hugging Face**](https://huggingface.co/spaces/premkondru/montage-agent)

Steps:

1. Adjust settings (before upload):
   * **Event name**
   * **Labels** (related to the event, e.g., *audience, stage, portraits*)
   * **Max images per carousel**
   * **Captioner mode:**
     * **Template Mode** → tighter, deterministic captions (good for campaigns).
     * **BLIP-2 (LoRA) Mode** → creative, abstract captions (good for storytelling).
2. Upload event photos (optional):
   Sample images are pre-provided in the demo, so for quick testing you don’t need to upload your own.
   You can also check “use only current upload session” if you want the app to ignore previously loaded images and restrict itself to just your current upload.
3. Preview Instagram-style carousels with captions and hashtags.
4. Exclude/include individual photos as needed.
5. Download/export JSON for Instagram publishing.

This is the fastest way to test the workflow without installing anything locally.

---

## ✨ Core Features

* **Task automated:** Raw photo dumps → IG-ready posts.
* **Fine-tuned model:** BLIP-2 (Flan-T5-XL) with LoRA (\~0.47% trainable params).
* **Why LoRA:** Specializes captioner in Montage Club style → abstract, clean, no noisy @handles.
* **Evaluation:** Validation loss, BLEU/ROUGE, CLIPScore + human ratings (IG-readiness).
* **Outcome:** Abstract, creative captions consistently loved by the club.
* **Streamlit UI:** Upload, dedupe, cluster & caption photos with human-in-the-loop edits.
* **Hybrid captioner:** Template for concise captions, BLIP-2 for creative captions.

---

## 📦 Deliverables

1. **[GitHub Repo – MontageAgent](https://github.com/premkondru/MontageAgent)** — Source code, configs, training scripts.
2. **[Hugging Face Space – montage-agent](https://huggingface.co/spaces/premkondru/montage-agent)** — Live demo Streamlit app.
3. **[Architecture (PDF)](deliverables/MontageAgent-AIAgentArchitecture.pdf)** — High-level design overview.
4. **[Data Science Report (PDF)](deliverables/MontageAgent-Data%20Science%20Report.pdf)** — Datasets, fine-tuning setup, evaluation, results.
5. **[Code Companion (PDF)](deliverables/MontageAgent-Code-ChatGPT.pdf)** — Code-focused explanations.
6. **[Architecture Doc – ChatGPT (PDF)](deliverables/MontageAgent-ArchitectureDoc-ChatGPT.pdf)**
7. **[Data Science Report – ChatGPT (PDF)](deliverables/MontageAgent-DataScienceReport-ChatGPT.pdf)**
8. **[Product Screenshots (PDF)](deliverables/MontageAgent-Screenshots.pdf)**
9. **[Screen Recording (MOV)](deliverables/MontageAgent-ScreenRecording.mov)** — Demo video.

---

## 🚀 Getting Started (Local Setup)

### 1. Clone the repo and create environment

```bash
git clone https://github.com/premkondru/MontageAgent.git
cd MontageAgent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Launch the GUI (Streamlit)

```bash
streamlit run ui/streamlit_app.py
```

### 4. Run the pipeline (CLI mode)

```bash
python -m agent.supervisor --config configs/agent.yaml
```

### 5. Train BLIP-2 LoRA (optional)

```bash
accelerate launch training/train_lora_blip2.py --config configs/lora_blip2.yaml
```

---

## 📂 Repository Layout

* **`agent/`** — Production pipeline (Supervisor + stage tools).
* **`configs/`** — YAML configs for agent pipeline and LoRA training.
* **`training/`** — Scripts for fine-tuning BLIP-2 with LoRA.
* **`checkpoints/`** — Saved model checkpoints.
* **`ui/`** — Streamlit GUI app.
* **`deliverables/`** — Reports, screenshots, videos, architecture docs.
* **`publisher/`** — Export/IG integration wrappers.
* **`data/`** — Example datasets (images + caption files).

---

## 🧠 Mental Model

* **`agent/`** = Production pipeline.
* **`training/` + `checkpoints/`** = Model improvement loop.
* **`configs/`** = Switches everything via YAML.
* **`ui/`** = Human-facing GUI.
* **`deliverables/`** = Outputs to share with stakeholders.

---

