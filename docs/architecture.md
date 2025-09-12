# Architecture

## Overview
- **Agents:** Planner/Supervisor → Workers (Ingest, DedupeQuality, Categorizer, Selector, Captioner, Publisher)
- **Pattern:** Planner–Executor with Critic loop; tool-centric design
- **Models:** CLIP embeddings (selection/categorization), BLIP-2 + LoRA (caption), RAG (style/hashtags)
- **Integrations:** Local FS, optional Instagram Graph API, optional MCP servers

## Flow
1. Ingest daywise folders → 2. De-dup & quality → 3. Categorize → 4. Select diverse set →
5. Caption (LoRA + RAG) → 6. Critic checks → 7. Publish (optional).

## Rationale
- Parameter-efficient LoRA adapts style reliably.
- CLIP embeddings enable zero-shot tags + diversity selection.
- RAG ensures club-specific voice & hashtag consistency.
