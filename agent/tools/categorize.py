# agent/tools/categorize.py
import torch, open_clip
import numpy as np
from typing import List, Dict
from agent.tools.clip_features import encode_paths

_DEFAULT_LABELS = [
    "candid", "portrait", "group photo", "stage", "audience",
    "speaker", "award", "sports", "food", "night", "landscape",
    "architecture", "indoors", "outdoors"
]

class Categorizer:
    def __init__(self, cfg):
        self.cfg = cfg
        emb = cfg.get("embeddings", {}) if isinstance(cfg, dict) else {}
        self.model_name = emb.get("model", "ViT-B-32")
        self.pretrained = emb.get("pretrained", "laion2b_s34b_b79k")
        self.device = emb.get("device", "cpu")
        self.labels: List[str] = cfg.get("categorize", {}).get("labels", _DEFAULT_LABELS)
        self.th = float(cfg.get("categorize", {}).get("threshold", 0.22))  # was 0.28
        self.topk = int(cfg.get("categorize", {}).get("topk", 4))          # was 3
        self.min_labels = int(cfg.get("categorize", {}).get("min_labels", 2))  # NEW

        # lazy-load text encoder
        self._txt_model = None
        self._tokenizer = None

    def _lazy_text(self):
        if self._txt_model is None or self._tokenizer is None:
            m, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained, device=self.device
            )
            m.eval()
            self._txt_model = m
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
        return self._txt_model, self._tokenizer

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        m, tok = self._lazy_text()
        with torch.no_grad():
            tokens = tok(texts).to(self.device)
            feats = m.encode_text(tokens)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats.cpu().numpy().astype("float32")

    def __call__(self, items: List[Dict]):
        if not items:
            return items

        # 1) encode images with existing utility
        paths = [it["path"] for it in items]
        img_feats = encode_paths(paths, self.model_name, self.pretrained, self.device)

        # 2) encode candidate labels as prompts (CLIP likes natural language)
        prompts = [f"a photo of {lab}" for lab in self.labels]
        txt_feats = self._encode_texts(prompts)  # [L, D]

        # 3) cosine similarity matrix [N, L]
        sims = img_feats @ txt_feats.T  # unit-norm → dot == cosine

        # 4) assign labels
        for i, it in enumerate(items):
            scores = sims[i]
            order = np.argsort(-scores)[: self.topk]
            chosen = [self.labels[j] for j in order if scores[j] >= self.th]
            # simple indoor/outdoor/night heuristics (optional)
            if "night" not in chosen:
                try:
                    from PIL import ImageStat, Image
                    im = Image.open(it["path"]).convert("L")
                    bright = ImageStat.Stat(im).mean[0]
                    if bright < 35:  # very dark
                        chosen.append("night")
                except Exception:
                    pass
            # ensure minimum number of labels by filling from the remaining best
            if len(chosen) < self.min_labels:
                for j in order:
                    lab = self.labels[j]
                    if lab not in chosen:
                        chosen.append(lab)
                    if len(chosen) >= self.min_labels:
                        break
            it["labels"] = chosen or ["candid"]
        return items
