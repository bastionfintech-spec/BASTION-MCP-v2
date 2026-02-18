#!/usr/bin/env python3
"""
BASTION v6 QLoRA Fine-Tune Script
==================================
Builds on v5 merged model with 38 new structure-aware multi-pair examples.
Total: 328 examples (290 v5 + 38 v6)

Run on GPU cluster (4x RTX 5090):
  1. Upload finetune_v6.py + merge_v6.py + bastion_risk_v6_combined.jsonl to /workspace/
  2. Stop vLLM: pkill -f vllm
  3. Run: /opt/sys-venv/bin/python3 /workspace/finetune_v6.py
  4. After completion, run /workspace/merge_v6.py separately
  5. Start vLLM with new model

All fixes from v3/v5 pre-applied.
"""

import os
import sys
import json
import time
import torch
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
BASE_MODEL = "/root/models/bastion-risk-v5-merged"  # Build on v5
TRAINING_DATA = "/workspace/bastion_risk_v6_combined.jsonl"
OUTPUT_DIR = "/workspace/bastion-risk-v6-lora"
MERGED_DIR = "/root/models/bastion-risk-v6-merged"

# QLoRA settings — same as v5
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
NUM_EPOCHS = 4
BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_LENGTH = 4096
WARMUP_RATIO = 0.05

print("=" * 60)
print("BASTION v6 QLoRA Fine-Tune")
print("=" * 60)
print(f"Base model: {BASE_MODEL}")
print(f"Training data: {TRAINING_DATA}")
print(f"Output: {OUTPUT_DIR}")
print(f"Merged output: {MERGED_DIR}")
print(f"Start time: {datetime.now()}")
print()

# ============================================================
# Verify prerequisites
# ============================================================
print("[1/5] Checking prerequisites...")

if not os.path.exists(BASE_MODEL):
    print(f"  ERROR: Base model not found at {BASE_MODEL}")
    for fallback in ["/root/models/bastion-risk-v3-merged", "/root/models/bastion-risk-merged"]:
        if os.path.exists(fallback):
            BASE_MODEL = fallback
            print(f"  Using fallback: {BASE_MODEL}")
            break
    else:
        print("  No model found! Exiting.")
        sys.exit(1)

if not os.path.exists(TRAINING_DATA):
    print(f"  ERROR: Training data not found at {TRAINING_DATA}")
    sys.exit(1)

with open(TRAINING_DATA, 'r', encoding='utf-8') as f:
    num_examples = sum(1 for line in f if line.strip())
print(f"  Training examples: {num_examples}")

if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {name} ({mem:.0f}GB)")
else:
    print("  ERROR: No GPU available!")
    sys.exit(1)

import subprocess
result = subprocess.run(["pgrep", "-f", "vllm"], capture_output=True, text=True)
if result.stdout.strip():
    print("  WARNING: vLLM is still running! Killing it...")
    subprocess.run(["pkill", "-f", "vllm"], capture_output=True)
    time.sleep(5)
    torch.cuda.empty_cache()

print("  All checks passed!")
print()

# ============================================================
# Load model with QLoRA
# ============================================================
print("[2/5] Loading model with QLoRA quantization...")
start_load = time.time()

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa",
    dtype=torch.bfloat16,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
print(f"  Load time: {time.time()-start_load:.1f}s")
print()

# ============================================================
# Prepare dataset
# ============================================================
print("[3/5] Preparing dataset...")

from datasets import Dataset

conversations = []
with open(TRAINING_DATA, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        conversations.append(obj)

def format_chat(example):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = Dataset.from_list(conversations)
dataset = dataset.map(format_chat, remove_columns=dataset.column_names)
print(f"  Dataset size: {len(dataset)} examples")

sample_tokens = tokenizer(dataset[0]["text"], return_tensors="pt")
print(f"  Sample token length: {sample_tokens.input_ids.shape[1]}")
print()

# ============================================================
# Training
# ============================================================
print("[4/5] Starting training...")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE} x {GRAD_ACCUM} grad accum = {BATCH_SIZE * GRAD_ACCUM} effective")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Max length: {MAX_LENGTH}")

from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,
    max_length=MAX_LENGTH,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    report_to="none",
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

train_start = time.time()
result = trainer.train()
train_time = time.time() - train_start

print(f"\n  Training complete!")
print(f"  Time: {train_time/60:.1f} minutes")
print(f"  Final loss: {result.training_loss:.4f}")
print()

# ============================================================
# Save LoRA adapter
# ============================================================
print("[5/5] Saving LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"  Saved to: {OUTPUT_DIR}")
print()

# ============================================================
# Summary
# ============================================================
total_time = time.time() - start_load
print("=" * 60)
print("FINE-TUNE COMPLETE")
print("=" * 60)
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Training examples: {num_examples}")
print(f"Final loss: {result.training_loss:.4f}")
print(f"LoRA adapter: {OUTPUT_DIR}")
print()
print("NEXT STEP: Run merge separately to avoid OOM:")
print(f"  /opt/sys-venv/bin/python3 /workspace/merge_v6.py")
