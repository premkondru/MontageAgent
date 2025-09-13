"""
LoRA training for BLIP-2 (Flan-T5) to match Montage caption style.

Usage:
  accelerate launch training/train_lora_blip2.py

Data format (JSONL):
  {"image":"eventX/IMG_0012.jpg","event":"IITG Orientation 2025","labels":["stage","audience"],"caption":"Highlights from IITG Orientation — stage vibes & cheering crowd. Swipe →"}
  {"image":"eventY/IMG_0103.jpg","event":"Inter-hostel Dance Finals","labels":["portrait","stage"],"caption":"Faces of the finals — grit, lights, and a whole lot of heart. Swipe →"}

Config:
  configs/lora_blip2.yaml
"""

import os, json
from pathlib import Path
from dataclasses import dataclass
from platform import processor
from typing import List, Dict, Any

import torch
from torch.utils.data import (
    Dataset,
    DataLoader
)
from PIL import Image

from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    get_scheduler,
    default_data_collator,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator


# --------------------------- Config ---------------------------

def load_config(yaml_path="configs/lora_blip2.yaml") -> Dict[str, Any]:
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.rows[idx]
        image = Image.open(ex.image_path).convert("RGB")

        # Instruction prompt guides the model toward IG-style captions, no hashtags.
        label_str = ", ".join(ex.labels) if ex.labels else "event moments"
        event_str = f"about '{ex.event}'" if ex.event else "for a college event"
        prompt = (
            f"Write a short Instagram caption for a photography club post {event_str}. "
            f"Focus on: {label_str}. Keep it natural and clean. No hashtags."
        )

        inputs = self.processor(
            images=image,
            text=prompt,
            padding=False,          # keep False if using the collator
            return_tensors="pt"
        )
        # leave labels without padding; the collator will handle it

        labels = self.processor.tokenizer(
            ex.caption,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        batch = {k: v.squeeze(0) for k, v in inputs.items()}
        batch["labels"] = labels.squeeze(0)
        return batch


# --------------------------- Trainer ---------------------------

def build_model_and_processor(base_model: str, quantization: str):
    """
    Load BLIP-2 base with optional 8/4-bit quantization for LoRA training.
    """
    device_map = {"": 0} if torch.cuda.is_available() else None
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # BitsAndBytes is optional — import only if requested
    load_in_8bit = quantization == "bnb_8bit"
    load_in_4bit = quantization == "bnb_4bit"
    if (load_in_8bit or load_in_4bit) and not torch.cuda.is_available():
        print("[warn] quantization requested but CUDA not available; loading full precision.")
        load_in_8bit = load_in_4bit = False

    processor = AutoProcessor.from_pretrained(base_model)
    model = Blip2ForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )

    if load_in_8bit or load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    return model, processor


def attach_lora(model: Blip2ForConditionalGeneration, lora_cfg: Dict[str, Any]):
    """
    Attach LoRA adapters to attention projections in the language model.
    Adjust target_modules to taste.
    """
    lcfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        target_modules=tuple(lora_cfg.get("target_modules", ["q", "k", "v", "o"])),
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model


def main():
    cfg = load_config()

    base_model = cfg.get("base_model", "Salesforce/blip2-flan-t5-xl")
    quantization = cfg.get("quantization", "bnb_8bit")
    train_cfg = cfg["train"]

    outdir = Path(train_cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    # Build model / processor
    model, processor = build_model_and_processor(base_model, quantization)
    # Make sure padding is defined (T5 uses <pad>)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    model = attach_lora(model, cfg.get("lora", {}))
   
    # Datasets
    train_ds = CaptionDataset(train_cfg["train_jsonl"], train_cfg["image_root"], processor, train_cfg["max_seq_len"])
    val_ds   = CaptionDataset(train_cfg["val_jsonl"],   train_cfg["image_root"], processor, train_cfg["max_seq_len"])

    # Accelerator
    accel = Accelerator(gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)))
    device = accel.device

    # Dataloaders
    collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=model,
        padding=True,                 # dynamic pad to longest in batch
        label_pad_token_id=-100,      # ignore padded labels in loss
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=int(train_cfg["per_device_train_batch_size"]),
        shuffle=True,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=int(train_cfg["per_device_eval_batch_size"]),
        shuffle=False,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    # Optimizer & scheduler
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
    sched = get_scheduler(
        "cosine",
        optimizer=optim,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )

    model, optim, train_dl, val_dl = accel.prepare(model, optim, train_dl, val_dl)

    # Training loop
    save_every = int(train_cfg.get("save_every_steps", 1000))
    step = 0
    best_val = None

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dl:
            with accel.accumulate(model):
                out = model(**batch)
                loss = out.loss
                accel.backward(loss)
                optim.step()
                sched.step()
                optim.zero_grad()

            if accel.is_main_process and step % 50 == 0:
                print(f"[train] step {step} loss {loss.item():.4f}")

            if accel.is_main_process and step > 0 and step % save_every == 0:
                accel.unwrap_model(model).save_pretrained(
                    str(outdir / f"step_{step}"), safe_serialization=True
                )

            step += 1
            if max_steps > 0 and step >= max_steps:
                break

        # Validation (quick loss)
        model.eval()
        vloss = 0.0
        vcnt = 0
        with torch.no_grad():
            for vb in val_dl:
                out = model(**vb)
                vloss += out.loss.item()
                vcnt += 1

        vloss = vloss / max(1, vcnt)
        if accel.is_main_process:
            print(f"[val] epoch {epoch} val_loss {vloss:.4f}")
            # Always save "last"
            accel.unwrap_model(model).save_pretrained(str(outdir / "last"), safe_serialization=True)
            # Save "best"
            if (best_val is None) or (vloss < best_val):
                best_val = vloss
                accel.unwrap_model(model).save_pretrained(str(outdir / "best"), safe_serialization=True)

    if accel.is_main_process:
        print("[done] training complete.")
        print(f"Adapters saved to: {outdir}")


if __name__ == "__main__":
    main()
