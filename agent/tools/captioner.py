# agent/tools/captioner.py
import os, json, math, random, itertools
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple

import numpy as np

from agent.tools.clip_features import encode_paths, encode_texts

class Captioner:
    """
    Intelligent, cluster-aware captioner.
    - Uses CLIP to retrieve stylistically similar past captions (RAG).
    - Builds clean, IG-ready captions (no inline hashtags).
    - Derives hashtags from base + label mapping + retrieved history.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.cap_cfg = (cfg.get("captioner") or {})
        self.emb_cfg = (cfg.get("embeddings") or {})
        # CLIP setup
        self.model = self.emb_cfg.get("model", "ViT-B-32")
        self.pretrained = self.emb_cfg.get("pretrained", "laion2b_s34b_b79k")
        self.device = self.emb_cfg.get("device", "cpu")

        # Behavior knobs
        self.mode = (self.cap_cfg.get("mode") or "template").lower()
        self.max_hashtags = int(self.cap_cfg.get("max_hashtags", 15))
        self.include_swipe_hint = bool(self.cap_cfg.get("include_swipe_hint", True))
        self.openers = self.cap_cfg.get("openers") or ["Highlights from"]
        self.base_tags = self.cap_cfg.get("base_hashtags") or ["#IITGuwahati","#Montage","#PhotographyClub"]
        self.label_tag_map: Dict[str, List[str]] = self.cap_cfg.get("label_hashtags") or {}

        # Past captions file for RAG
        self.past_path = self.cap_cfg.get("past_captions_path") or "data/style/past_captions.jsonl"
        self._past = self._load_past(self.past_path)  # list of dicts: {caption, hashtags, image?}
        self._past_texts = [p.get("caption","") for p in self._past]
        # Pre-embed past captions for fast retrieval
        self._past_txt_emb = encode_texts(self._past_texts, self.model, self.pretrained, self.device) if self._past_texts else None

    # ---------- public API ----------
    def __call__(self, clusters: List[Dict[str, Any]], cluster_mode: bool = True) -> List[Dict[str, Any]]:
        posts = []
        for cl in clusters:
            paths = [m["path"] for m in cl["items"]]
            event_name = self._derive_event_name(cl["items"])
            top_labels = self._aggregate_labels(cl["items"], topk=3)

            # RAG: fetch stylistic hints from past captions
            hints = self._retrieve_style_hints(paths, k=3)

            # Build caption
            if self.mode == "blip2":
                # (Optional) Hook for a BLIP-2 LoRA captioner; falls back to template if not available.
                caption = self._caption_via_blip2(paths, event_name, top_labels, hints) or \
                          self._caption_via_template(event_name, top_labels, hints)
            else:
                caption = self._caption_via_template(event_name, top_labels, hints)

            # Derive hashtags
            hashtags = self._build_hashtags(top_labels, hints)
            posts.append({
                "images": paths,
                "caption": caption,
                "hashtags": hashtags,
                "labels": list(top_labels),   # expose for UI
                "cluster_id": int(cl.get("cluster_id", 0))
            })
        return posts

    # ---------- caption building ----------
    def _caption_via_template(self, event: str, labels: List[str], hints: Dict[str, Any]) -> str:
        opener = random.choice(self.openers) if self.openers else "Highlights from"
        label_phrase = self._humanize_labels(labels)
        pieces = [f"{opener} {event}" if event else opener]
        if label_phrase:
            pieces.append(f"— {label_phrase}.")
        if self.include_swipe_hint:
            pieces.append("Swipe →")
        # Style borrow: if we have a standout fragment, append softly
        tail = hints.get("style_tail")
        if tail:
            pieces.append(tail)
        # Join and clean spacing
        text = " ".join(pieces).replace("  ", " ").strip()
        return text

    def _caption_via_blip2(self, paths: List[str], event: str, labels: List[str], hints: Dict[str, Any]) -> str:
        """
        Placeholder for a BLIP-2 LoRA model (see configs/lora_blip2.yaml).
        Return None to fall back to template if transformers pipeline not available.
        """
        try:
            import torch
            from transformers import Blip2ForConditionalGeneration, AutoProcessor
        except Exception:
            return None  # transformers not installed

        # NOTE: This is a lightweight sketch; you can load your LoRA-adapted weights here.
        # For now, prefer the template for speed/reliability unless you add the model.
        return None

    # ---------- hashtags ----------
    def _build_hashtags(self, labels: List[str], hints: Dict[str, Any]) -> List[str]:
        tags = list(self.base_tags)  # start with core tags
        # Label-driven tags
        for lab in labels:
            for t in self.label_tag_map.get(lab, []):
                if t not in tags:
                    tags.append(t)
        # Retrieved historical tags (keep top ones)
        top_hist = hints.get("top_hist_tags", [])
        for t in top_hist:
            if len(tags) >= self.max_hashtags:
                break
            if t not in tags:
                tags.append(t)
        # Dedup + cap
        seen = set()
        deduped = []
        for t in tags:
            if t.startswith("#") and t.lower() not in seen:
                seen.add(t.lower())
                deduped.append(t)
            if len(deduped) >= self.max_hashtags:
                break
        return deduped

    # ---------- retrieval (RAG) ----------
    def _retrieve_style_hints(self, paths: List[str], k: int = 3) -> Dict[str, Any]:
        if not self._past or self._past_txt_emb is None:
            return {}
        # Represent the cluster by mean image embedding
        img_emb = encode_paths(paths, self.model, self.pretrained, self.device)  # [N,D]
        if img_emb.size == 0:
            return {}
        cluster_vec = (img_emb.mean(axis=0, keepdims=True) / (np.linalg.norm(img_emb.mean(axis=0))+1e-8)).astype("float32")  # [1,D]
        sims = (cluster_vec @ self._past_txt_emb.T).ravel()  # cosine because both are unit
        order = np.argsort(-sims)[:k]
        top_caps = [self._past[i].get("caption","") for i in order]
        top_tags = list(itertools.chain.from_iterable(self._past[i].get("hashtags", []) for i in order))
        # Build a small stylistic tail (e.g., a short phrase to append)
        style_tail = self._pick_tail(top_caps)
        # Count historical tags popularity
        tag_counts = Counter([t for t in top_tags if isinstance(t,str) and t.startswith("#")])
        top_hist_tags = [t for t,_ in tag_counts.most_common(6)]
        return {"style_tail": style_tail, "top_hist_tags": top_hist_tags}

    def _pick_tail(self, caps: List[str]) -> str:
        """
        Extract a short, soft stylistic phrase if available.
        Example: 'memories', 'club vibes', 'golden hour'.
        """
        # naive heuristic: keep short <= 3-word trailing fragments without '#'
        candidates = []
        for c in caps:
            c = (c or "").strip()
            if not c: continue
            parts = [p.strip() for p in c.split("—")] if "—" in c else [c]
            tail = parts[-1]
            if "#" in tail:  # drop hashtag-only tails
                continue
            # keep 2-3 word tails
            words = tail.split()
            if 1 <= len(words) <= 4:
                candidates.append(tail)
        return random.choice(candidates) if candidates else ""

    # ---------- utilities ----------
    def _derive_event_name(self, items: List[Dict[str, Any]]) -> str:
        """
        Use the most common 'day' or parent folder name to form an event label.
        """
        day_vals = [it.get("day") for it in items if it.get("day")]
        if day_vals:
            day = Counter(day_vals).most_common(1)[0][0]
            return str(day).replace("_"," ").replace("-"," ").strip()
        # fallback to parent dir of first image
        p0 = items[0]["path"]
        parent = os.path.basename(os.path.dirname(p0))
        return parent.replace("_"," ").replace("-"," ").strip()

    def _aggregate_labels(self, items: List[Dict[str, Any]], topk: int = 3) -> List[str]:
        cnt = Counter()
        for it in items:
            for lab in it.get("labels", []) or []:
                cnt[lab] += 1
        return [lab for lab,_ in cnt.most_common(topk)]

    def _humanize_labels(self, labels: List[str]) -> str:
        if not labels: return ""
        if len(labels) == 1: return labels[0]
        if len(labels) == 2: return f"{labels[0]} & {labels[1]}"
        return f"{labels[0]}, {labels[1]} & {labels[2]}"

    def _load_past(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path): return []
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        return rows
