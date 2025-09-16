"""
LoRA training for BLIP-2 (Flan-T5) with BLEU/ROUGE/CLIPScore evaluation.
Supports CUDA, Apple Silicon (MPS), and CPU. Includes verbose diagnostics
for CLIPScore availability and NaN guards for stability.

Usage:
  accelerate launch training/train_lora_blip2.py
"""

import os
import re
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    get_scheduler,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator

# --------------------------- Config ---------------------------

def load_config(yaml_path: str = "configs/lora_blip2.yaml") -> Dict[str, Any]:
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# --------------------------- Dataset ---------------------------

@dataclass
class Example:
    image_path: str
    caption: str
    event: str
    labels: List[str]

class CaptionDataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: str, processor: AutoProcessor, max_seq_len: int = 96):
        self.rows: List[Example] = []
        self.processor = processor
        self.image_root = Path(image_root)
        self.max_seq_len = max_seq_len

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                img = self.image_root / r["image"]
                cap = r.get("caption")
                if img.exists() and cap:
                    self.rows.append(
                        Example(
                            image_path=str(img),
                            caption=cap,
                            event=r.get("event", ""),
                            labels=r.get("labels", []),
                        )
                    )

    def __len__(self) -> int:
        return len(self.rows)

    def _make_prompt(self, ex: Example) -> str:
        label_str = ", ".join(ex.labels) if ex.labels else "event moments"
        event_str = f"about '{ex.event}'" if ex.event else "for a college event"
        return (
            f"Write a short Instagram caption for a photography club post {event_str}. "
            f"Focus on: {label_str}. Keep it natural and clean."
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.rows[idx]
        image = Image.open(ex.image_path).convert("RGB")
        prompt = self._make_prompt(ex)

        inputs = self.processor(images=image, text=prompt, padding=False, return_tensors="pt")

        labels_ids = self.processor.tokenizer(
            ex.caption,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        if torch.is_tensor(labels_ids):
            labels_tensor = labels_ids.detach().clone().to(dtype=torch.long)
        else:
            labels_tensor = torch.tensor(labels_ids, dtype=torch.long)

        batch: Dict[str, Any] = {k: v.squeeze(0) for k, v in inputs.items()}
        batch["labels"] = labels_tensor
        batch["ref_caption"] = ex.caption
        batch["image_path_str"] = ex.image_path
        return batch

# --------------------------- Text Normalization ---------------------------

_PUNC_RE = re.compile(r"[^\w\s#]+", flags=re.UNICODE)

def normalize_text(
    s: str,
    lowercase: bool = True,
    strip_punct: bool = True,
    remove_hashtags: bool = False,
    remove_swipe_tokens: bool = True,
) -> str:
    if lowercase:
        s = s.lower()
    s = s.strip()
    if remove_swipe_tokens:
        s = s.replace("swipe →", " ").replace("swipe->", " ").replace("swipe right", " ")
    if remove_hashtags:
        s = re.sub(r"#\w+", " ", s)
    if strip_punct:
        s = _PUNC_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# --------------------------- Metrics ---------------------------

def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def bleu_generic(tokens_pred: List[str], tokens_ref: List[str], N: int) -> float:
    if not tokens_pred or not tokens_ref:
        return 0.0
    weights = [1.0 / N] * N
    precisions = []
    for n in range(1, N + 1):
        p_ngrams = _ngrams(tokens_pred, n)
        r_ngrams = set(_ngrams(tokens_ref, n))
        if len(p_ngrams) == 0:
            precisions.append(1e-9)
            continue
        match = sum(1 for g in p_ngrams if g in r_ngrams)
        precisions.append(max(match, 1e-9) / len(p_ngrams))
    c, r = len(tokens_pred), len(tokens_ref)
    bp = 1.0 if c > r else math.exp(1 - r / max(c, 1))
    return float(bp * math.exp(sum(w * math.log(p) for w, p in zip(weights, precisions))))

def bleu1(tp, tr): return bleu_generic(tp, tr, 1)
def bleu2(tp, tr): return bleu_generic(tp, tr, 2)
def bleu4(tp, tr): return bleu_generic(tp, tr, 4)

def rougeL_lcs_f1(tokens_pred: List[str], tokens_ref: List[str]) -> float:
    m, n = len(tokens_ref), len(tokens_pred)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            dp[i+1][j+1] = dp[i][j] + 1 if tokens_ref[i] == tokens_pred[j] else max(dp[i][j+1], dp[i+1][j])
    lcs = dp[m][n]
    prec = lcs / max(n, 1)
    rec  = lcs / max(m, 1)
    beta2 = 1.2**2
    denom = prec + beta2*rec
    return (1+beta2)*prec*rec/denom if denom > 0 else 0.0

# --------------------------- Optional CLIPScore ---------------------------

def maybe_load_openclip(clip_cfg: Dict[str, Any], device: torch.device):
    if not clip_cfg or not clip_cfg.get("enabled", False):
        print("[clipscore] disabled in config (set clipscore.enabled: true)")
        return None
    try:
        import open_clip
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            clip_cfg.get("model", "ViT-B-32"),
            pretrained=clip_cfg.get("pretrained", "laion2b_s34b_b79k"),
            device=device_str
        )
        tokenizer = open_clip.get_tokenizer(clip_cfg.get("model", "ViT-B-32"))
        model.eval()
        print(f"[clipscore] OpenCLIP ready on {device_str}")
        return (model, preprocess, tokenizer)
    except Exception as e:
        print(f"[clipscore] FAILED to load OpenCLIP: {e}")
        return None

@torch.no_grad()
def compute_clipscore_openclip(image_paths: List[str], texts: List[str], clip_bundle, device: torch.device) -> Optional[float]:
    if clip_bundle is None:
        return None
    model, preprocess, tokenizer = clip_bundle
    ims = []
    bad = 0
    for p in image_paths:
        try:
            ims.append(preprocess(Image.open(p).convert("RGB")).unsqueeze(0))
        except Exception as e:
            print(f"[clipscore] could not open image {p}: {e}")
            bad += 1
            continue
    if not ims:
        print("[clipscore] no valid images in batch (all failed to open?)")
        return None
    imgs = torch.cat(ims, dim=0)
    try:
        txt_tokens = tokenizer(texts)
    except TypeError:
        txt_tokens = tokenizer(texts, truncate=True)
    if not torch.is_tensor(txt_tokens):
        txt_tokens = torch.tensor(txt_tokens)
    img_emb = model.encode_image(imgs)
    txt_emb = model.encode_text(txt_tokens)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    sims = (img_emb * txt_emb).sum(dim=-1)
    return sims.mean().item()

# --------------------------- Device selection ---------------------------

def resolve_device_map_and_dtype(quantization: str):
    use_cuda = torch.cuda.is_available()
    use_mps = (not use_cuda) and torch.backends.mps.is_available()
    if use_cuda:
        return {"": 0}, torch.bfloat16, (quantization == "bnb_8bit"), (quantization == "bnb_4bit")
    elif use_mps:
        print("[info] Using Apple Silicon GPU (MPS)")
        if quantization in ("bnb_8bit", "bnb_4bit"):
            print("[warn] bitsandbytes not supported on MPS; using full precision.")
        return {"": "mps"}, torch.float32, False, False
    else:
        print("[info] Using CPU")
        return None, torch.float32, False, False

def build_model_and_processor(base_model: str, quantization: str):
    device_map, dtype, load_in_8bit, load_in_4bit = resolve_device_map_and_dtype(quantization)
    processor = AutoProcessor.from_pretrained(base_model, use_fast=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        base_model,
        dtype=dtype,
        device_map=device_map,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )
    if load_in_8bit or load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = processor.tokenizer.pad_token_id
    return model, processor

def attach_lora(model: Blip2ForConditionalGeneration, lora_cfg: Dict[str, Any]):
    lcfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        target_modules=tuple(lora_cfg.get("target_modules", ["q","k","v","o"])),
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model

# --------------------------- Collator ---------------------------

class MetaAwareSeq2SeqCollator:
    def __init__(self, tokenizer, model, label_pad_token_id=-100, padding=True):
        self.base = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=padding,
            label_pad_token_id=label_pad_token_id,
        )
        self.meta_keys = ("ref_caption", "image_path_str")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        metas = {k: [f.get(k, None) for f in features] for k in self.meta_keys}
        stripped = [{k: v for k, v in f.items() if k not in self.meta_keys} for f in features]
        batch = self.base(stripped)
        for k in self.meta_keys:
            batch[k] = metas[k]
        return batch

# --------------------------- Helpers ---------------------------

ALLOWED_MODEL_KEYS = {"pixel_values", "input_ids", "attention_mask", "labels"}
def to_model_inputs(batch: dict) -> dict:
    return {k: v for k, v in batch.items() if k in ALLOWED_MODEL_KEYS}

# --------------------------- Validation ---------------------------

@torch.no_grad()
def run_validation_loss(model, val_dl, accel) -> float:
    model.eval()
    vloss = 0.0
    vcnt = 0
    for vb in val_dl:
        out = model(**to_model_inputs(vb))
        vloss += out.loss.item()
        vcnt += 1
    vloss = vloss / max(1, vcnt)
    if accel.is_main_process:
        print(f"[val] loss {vloss:.4f}")
    return vloss

@torch.no_grad()
def run_validation_metrics(model, processor, val_dl, accel, eval_cfg: Dict[str, Any], clip_bundle) -> Dict[str, float]:
    if not eval_cfg.get("generate", True):
        return {}
    num_beams = int(eval_cfg.get("num_beams", 4))
    max_new_tokens = int(eval_cfg.get("max_new_tokens", 48))
    length_penalty = float(eval_cfg.get("length_penalty", 0.9))
    no_repeat = int(eval_cfg.get("no_repeat_ngram_size", 3))
    subset_size = int(eval_cfg.get("eval_subset_size", -1))
    norm_cfg = eval_cfg.get("normalize", {})
    lowercase = bool(norm_cfg.get("lowercase", True))
    strip_punct = bool(norm_cfg.get("strip_punct", True))
    remove_hashtags = bool(norm_cfg.get("remove_hashtags", False))
    remove_swipe = bool(norm_cfg.get("remove_swipe_tokens", True))

    bleu1_list: List[float] = []
    bleu2_list: List[float] = []
    bleu4_list: List[float] = []
    rouge_list: List[float] = []
    clip_scores: List[float] = []

    count = 0
    for batch in val_dl:
        inputs = {k: v for k, v in batch.items() if k in ("pixel_values", "input_ids", "attention_mask")}
        gen_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            do_sample=False,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat,
        )
        preds = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        refs = batch["ref_caption"]
        img_paths = batch["image_path_str"]

        for pred, ref in zip(preds, refs):
            pred_n = normalize_text(pred, lowercase, strip_punct, remove_hashtags, remove_swipe)
            ref_n  = normalize_text(ref,  lowercase, strip_punct, remove_hashtags, remove_swipe)
            pt = pred_n.split()
            rt = ref_n.split()
            bleu1_list.append(bleu1(pt, rt))
            bleu2_list.append(bleu2(pt, rt))
            bleu4_list.append(bleu4(pt, rt))
            rouge_list.append(rougeL_lcs_f1(pt, rt))

        cs = compute_clipscore_openclip(img_paths, preds, clip_bundle, accel.device) if clip_bundle is not None else None
        if cs is not None:
            clip_scores.append(cs)

        count += len(preds)
        if subset_size > 0 and count >= subset_size:
            break

    metrics: Dict[str, float] = {}
    if bleu1_list: metrics["bleu1_mean"] = sum(bleu1_list)/len(bleu1_list)
    if bleu2_list: metrics["bleu2_mean"] = sum(bleu2_list)/len(bleu2_list)
    if bleu4_list: metrics["bleu4_mean"] = sum(bleu4_list)/len(bleu4_list)
    if rouge_list: metrics["rougeL_mean"] = sum(rouge_list)/len(rouge_list)
    if clip_scores: metrics["clipscore_mean"] = sum(clip_scores)/len(clip_scores)

    if accel.is_main_process:
        parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
        if "clipscore_mean" not in metrics:
            parts.append("clipscore_mean=DISABLED/UNAVAILABLE (enable in YAML and install open-clip-torch)")
        print(f"[val] gen-metrics: " + (", ".join(parts) if parts else "(none)"))
    return metrics

# --------------------------- Main ---------------------------

def main():
    cfg = load_config()
    base_model = cfg.get("base_model", "Salesforce/blip2-flan-t5-xl")
    quantization = cfg.get("quantization", "none")
    train_cfg = cfg["train"]
    eval_cfg = cfg.get("eval", {"generate": True})
    clip_cfg = cfg.get("clipscore", {"enabled": False})

    outdir = Path(train_cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    model, processor = build_model_and_processor(base_model, quantization)
    model = attach_lora(model, cfg.get("lora", {}))

    accel = Accelerator(gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)))
    device = accel.device

    train_ds = CaptionDataset(train_cfg["train_jsonl"], train_cfg["image_root"], processor, train_cfg["max_seq_len"])
    val_ds   = CaptionDataset(train_cfg["val_jsonl"],   train_cfg["image_root"], processor, train_cfg["max_seq_len"])

    clip_bundle = maybe_load_openclip(clip_cfg, device)
    if accel.is_main_process:
        if clip_cfg.get("enabled", False) and clip_bundle is None:
            print("[info] CLIPScore requested but NOT ACTIVE — see [clipscore] logs above.")
        elif clip_bundle is not None:
            print("[info] CLIPScore ENABLED.")

    collator = MetaAwareSeq2SeqCollator(tokenizer=processor.tokenizer, model=model)
    train_dl = DataLoader(train_ds, batch_size=int(train_cfg["per_device_train_batch_size"]),
                          shuffle=True, collate_fn=collator)
    val_dl   = DataLoader(val_ds,   batch_size=int(train_cfg["per_device_eval_batch_size"]),
                          shuffle=False, collate_fn=collator)

    lr = float(train_cfg.get("lr", 2e-4))
    wd = float(train_cfg.get("weight_decay", 0.01))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    steps_per_epoch = max(1, len(train_dl))
    max_steps = int(train_cfg.get("max_steps", -1))
    if max_steps > 0:
        total_steps = max_steps
        num_epochs = 1
    else:
        num_epochs = int(train_cfg["num_train_epochs"])
        total_steps = steps_per_epoch * num_epochs

    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.05))
    sched = get_scheduler("cosine", optimizer=optim,
                          num_warmup_steps=int(total_steps*warmup_ratio),
                          num_training_steps=total_steps)

    model, optim, train_dl, val_dl = accel.prepare(model, optim, train_dl, val_dl)

    save_every = int(train_cfg.get("save_every_steps", 1000))
    step = 0
    best_val = None

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dl:
            with accel.accumulate(model):
                out = model(**to_model_inputs(batch))
                loss = out.loss
                if torch.isnan(loss):
                    print(f"[warn] NaN loss at step {step}; skipping update.")
                    optim.zero_grad()
                    continue
                accel.backward(loss)
                optim.step()
                sched.step()
                optim.zero_grad()

            if accel.is_main_process and step % 50 == 0:
                print(f"[train] step {step} loss {loss.item():.4f}")
            if accel.is_main_process and step > 0 and step % save_every == 0:
                accel.unwrap_model(model).save_pretrained(str(outdir/f"step_{step}"), safe_serialization=True)
            step += 1
            if max_steps > 0 and step >= max_steps:
                break

        vloss = run_validation_loss(model, val_dl, accel)
        metrics = run_validation_metrics(model, processor, val_dl, accel, eval_cfg, clip_bundle)
        monitor_val = metrics.get("rougeL_mean", vloss)
        if accel.is_main_process:
            accel.unwrap_model(model).save_pretrained(str(outdir/"last"), safe_serialization=True)
            if (best_val is None) or (monitor_val > best_val):
                best_val = monitor_val
                accel.unwrap_model(model).save_pretrained(str(outdir/"best"), safe_serialization=True)
                print(f"[ckpt] new best at epoch {epoch} (monitor score={monitor_val:.4f})")

    if accel.is_main_process:
        print("[done] training complete.")
        print(f"Adapters saved to: {outdir}")

if __name__ == "__main__":
    main()
