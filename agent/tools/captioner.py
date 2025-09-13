# agent/tools/captioner.py
import os, json, random, itertools
from collections import Counter
from typing import Dict, List, Any

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

        # CLIP setup (for retrieval + CLIPScore)
        self.model = self.emb_cfg.get("model", "ViT-B-32")
        self.pretrained = self.emb_cfg.get("pretrained", "laion2b_s34b_b79k")
        self.device = self.emb_cfg.get("device", "cpu")

        # --- LoRA/BLIP-2 captioner knobs ---
        self.mode = (self.cap_cfg.get("mode") or "template").lower()
        # BLIP-2 base model (separate from CLIP!)
        self.blip2_base = (
            self.cap_cfg.get("base_model")
            or self.cfg.get("base_model")
            or "Salesforce/blip2-flan-t5-xl"
        )
        self.adapter_path = (
            (self.cfg.get("captioner", {}) or {}).get("adapter_path")
            or (self.cfg.get("infer", {}) or {}).get("adapter_path")
            or "checkpoints/lora_blip2_montage/best"
        )
        self._blip = None  # lazy cache: (processor, model)

        # Caption/hashtag behavior
        self.max_hashtags = int(self.cap_cfg.get("max_hashtags", 15))
        self.include_swipe_hint = bool(self.cap_cfg.get("include_swipe_hint", True))
        self.openers = self.cap_cfg.get("openers") or ["Highlights from"]
        self.base_tags = self.cap_cfg.get("base_hashtags") or ["#IITGuwahati", "#Montage", "#PhotographyClub"]
        self.label_tag_map: Dict[str, List[str]] = self.cap_cfg.get("label_hashtags") or {}

        # Past captions (RAG)
        self.past_path = self.cap_cfg.get("past_captions_path") or "data/style/past_captions.jsonl"
        self._past = self._load_past(self.past_path)
        self._past_texts = [p.get("caption", "") for p in self._past]
        self._past_txt_emb = encode_texts(self._past_texts, self.model, self.pretrained, self.device) if self._past_texts else None

        # CLIPScore toggle
        self.calc_clipscore = bool(self.cap_cfg.get("calc_clipscore", True))

    # ---------- public API ----------
    def __call__(self, clusters: List[Dict[str, Any]], cluster_mode: bool = True) -> List[Dict[str, Any]]:
        posts = []
        for cl in clusters:
            paths = [m["path"] for m in cl["items"]]
            event_name = self._derive_event_name(cl["items"])
            top_labels = self._aggregate_labels(cl["items"], topk=3)

            # RAG: stylistic hints from past captions
            hints = self._retrieve_style_hints(paths, k=3)

            # Build caption
            if self.mode == "blip2":
                caption = self._caption_via_blip2(paths, event_name, top_labels, hints) or \
                          self._caption_via_template(event_name, top_labels, hints)
            else:
                caption = self._caption_via_template(event_name, top_labels, hints)

            # Hashtags
            hashtags = self._build_hashtags(top_labels, hints)

            # CLIPScore
            clipscore = self._clipscore(paths, caption) if self.calc_clipscore else None

            posts.append({
                "images": paths,
                "caption": caption,
                "hashtags": hashtags,
                "labels": list(top_labels),
                "cluster_id": int(cl.get("cluster_id", 0)),
                "clipscore": clipscore,
            })

        # Aggregate CLIPScores across posts (for the debug panel)
        means = [p["clipscore"]["mean"] for p in posts if p.get("clipscore") and p["clipscore"].get("mean") is not None]
        if means:
            means = np.array(means, dtype="float32")
            self.last_metrics = {
                "clipscore_mean": float(round(float(means.mean()), 4)),
                "clipscore_median": float(round(float(np.median(means)), 4)),
                "clipscore_min": float(round(float(means.min()), 4)),
                "clipscore_max": float(round(float(means.max()), 4)),
            }
        else:
            self.last_metrics = {"clipscore_mean": None}
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
        tail = hints.get("style_tail")
        if tail:
            pieces.append(tail)
        return " ".join(pieces).replace("  ", " ").strip()

    # ---------- BLIP-2 / LoRA ----------
    def _load_blip2_lora(self):
        # Robust import; return None on failure and let caller fall back
        try:
            import torch
            from transformers import AutoProcessor, Blip2ForConditionalGeneration
            from peft import PeftModel
        except Exception:
            return None

        try:
            base = self.blip2_base
            processor = AutoProcessor.from_pretrained(base)
            base_model = Blip2ForConditionalGeneration.from_pretrained(
                base,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map={"": 0} if torch.cuda.is_available() else None,
                load_in_8bit=False,
                load_in_4bit=False,
            )
            if os.path.isdir(self.adapter_path):
                model = PeftModel.from_pretrained(base_model, self.adapter_path)
            else:
                model = base_model
            model.eval()
            self._blip = (processor, model)
            return self._blip
        except Exception:
            return None

    def _caption_via_blip2(self, paths, event, labels, hints):
        try:
            from PIL import Image
        except Exception:
            return None

        if self._blip is None:
            self._blip = self._load_blip2_lora()
        if self._blip is None:
            return None

        processor, model = self._blip
        try:
            img_path = paths[len(paths) // 2]
            image = Image.open(img_path).convert("RGB")
            label_str = ", ".join(labels) if labels else "event moments"
            style_tail = hints.get("style_tail", "")
            prompt = (
                f"Write a short Instagram caption for a college photography club post about '{event}'. "
                f"Focus on: {label_str}. Keep it natural and clean. No hashtags. {style_tail}"
            ).strip()

            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
            gen = model.generate(
                **inputs,
                max_new_tokens=int(self.cfg.get("infer", {}).get("max_new_tokens", 48)),
                temperature=float(self.cfg.get("infer", {}).get("temperature", 0.7)),
                top_p=float(self.cfg.get("infer", {}).get("top_p", 0.9)),
                do_sample=True,
            )
            text = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
            if self.cfg.get("infer", {}).get("no_hashtags", True):
                text = " ".join([w for w in text.split() if not w.startswith("#")]).strip()
            return text
        except Exception:
            return None

    # ---------- hashtags ----------
    def _build_hashtags(self, labels: List[str], hints: Dict[str, Any]) -> List[str]:
        tags = list(self.base_tags)
        for lab in labels:
            for t in self.label_tag_map.get(lab, []):
                if t not in tags:
                    tags.append(t)
        top_hist = hints.get("top_hist_tags", [])
        for t in top_hist:
            if len(tags) >= self.max_hashtags:
                break
            if t not in tags:
                tags.append(t)
        seen = set(); deduped = []
        for t in tags:
            if t.startswith("#") and t.lower() not in seen:
                seen.add(t.lower()); deduped.append(t)
            if len(deduped) >= self.max_hashtags:
                break
        return deduped

    # ---------- retrieval (RAG) ----------
    def _retrieve_style_hints(self, paths: List[str], k: int = 3) -> Dict[str, Any]:
        if not self._past or self._past_txt_emb is None:
            return {}
        img_emb = encode_paths(paths, self.model, self.pretrained, self.device)
        if img_emb.size == 0:
            return {}
        mean = img_emb.mean(axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-8)
        cluster_vec = mean.reshape(1, -1).astype("float32")
        sims = (cluster_vec @ self._past_txt_emb.T).ravel()
        order = np.argsort(-sims)[:k]
        top_caps = [self._past[i].get("caption", "") for i in order]
        top_tags = list(itertools.chain.from_iterable(self._past[i].get("hashtags", []) for i in order))
        style_tail = self._pick_tail(top_caps)
        tag_counts = Counter([t for t in top_tags if isinstance(t, str) and t.startswith("#")])
        top_hist_tags = [t for t, _ in tag_counts.most_common(6)]
        return {"style_tail": style_tail, "top_hist_tags": top_hist_tags}

    def _pick_tail(self, caps: List[str]) -> str:
        candidates = []
        for c in caps:
            c = (c or "").strip()
            if not c:
                continue
            parts = [p.strip() for p in c.split("—")] if "—" in c else [c]
            tail = parts[-1]
            if "#" in tail:
                continue
            words = tail.split()
            if 1 <= len(words) <= 4:
                candidates.append(tail)
        return random.choice(candidates) if candidates else ""

    # ---------- utilities ----------
    def _derive_event_name(self, items: List[Dict[str, Any]]) -> str:
        day_vals = [it.get("day") for it in items if it.get("day")]
        if day_vals:
            day = Counter(day_vals).most_common(1)[0][0]
            return str(day).replace("_", " ").replace("-", " ").strip()
        p0 = items[0]["path"]
        parent = os.path.basename(os.path.dirname(p0))
        return parent.replace("_", " ").replace("-", " ").strip()

    def _aggregate_labels(self, items: List[Dict[str, Any]], topk: int = 3) -> List[str]:
        cnt = Counter()
        for it in items:
            for lab in it.get("labels", []) or []:
                cnt[lab] += 1
        return [lab for lab, _ in cnt.most_common(topk)]

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

    def _clipscore(self, paths, caption: str):
        try:
            if not caption or not paths:
                return None
            img_emb = encode_paths(paths, self.model, self.pretrained, self.device)
            if img_emb.size == 0:
                return None
            txt_emb = encode_texts([caption], self.model, self.pretrained, self.device)[0]
            per_image = (img_emb @ txt_emb).astype("float32")
            mean_dot = float(per_image.mean())
            mean_img = img_emb.mean(axis=0)
            mean_img = mean_img / (np.linalg.norm(mean_img) + 1e-8)
            mean_img_dot = float(mean_img @ txt_emb)
            return {
                "per_image": per_image.tolist(),
                "mean": round(mean_dot, 4),
                "mean_img": round(mean_img_dot, 4),
            }
        except Exception:
            return None
