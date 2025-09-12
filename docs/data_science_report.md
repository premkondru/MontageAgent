# Data Science Report

## Finetuning Setup
- **Data:** Past Montage captions/hashtags + new curated pairs.
- **Model:** BLIP-2 base with LoRA adapters (r=16, alpha=32, dropout=0.05).
- **Training:** 3–5 epochs, cosine LR, early stopping on validation CLIPScore/CIDEr.
- **Preprocessing:** Resize/center-crop images; normalize; JSONL with image path + caption.

## Evaluation
- **Categorization:** macro-F1/AP@K (labeled split).
- **Aesthetics/Quality:** correlation (Kendall-τ) with human ranking.
- **Captions:** CLIPScore; A/B human preference (target ≥60% wins).
- **End-to-end:** Ready-to-post rate ≥70%.

## Results (template)
- Table: Base vs LoRA metrics
- Ablations: +RAG vs no-RAG; selector variants (k-medoids vs random).

## Error Analysis
- Overly generic captions; wrong subject; tone drift. Mitigate with more in-domain pairs and stronger RAG constraints.
