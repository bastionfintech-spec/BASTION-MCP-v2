#!/usr/bin/env python3
"""
BASTION v6 LoRA Merge Script
==============================
Run AFTER finetune_v6.py completes (separate process to avoid OOM).
  /opt/sys-venv/bin/python3 /workspace/merge_v6.py
"""

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "/root/models/bastion-risk-v5-merged"
LORA_DIR = "/workspace/bastion-risk-v6-lora"
MERGED_DIR = "/root/models/bastion-risk-v6-merged"

# Check fallbacks
if not os.path.exists(BASE_MODEL):
    for fallback in ["/root/models/bastion-risk-v3-merged", "/root/models/bastion-risk-merged"]:
        if os.path.exists(fallback):
            BASE_MODEL = fallback
            break

print(f"Base model: {BASE_MODEL}")
print(f"LoRA adapter: {LORA_DIR}")
print(f"Output: {MERGED_DIR}")
print()

print("Loading base model in bf16...")
start = time.time()

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
print(f"  Loaded in {time.time()-start:.1f}s")

print("Merging LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_DIR)
model = model.merge_and_unload()
print("  Merged!")

print(f"Saving to {MERGED_DIR}...")
os.makedirs(MERGED_DIR, exist_ok=True)
model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

total = time.time() - start
print(f"Done! Total time: {total/60:.1f} minutes")
print()
print("NEXT: Start vLLM with:")
print(f"  nohup /opt/sys-venv/bin/python3 -m vllm.entrypoints.openai.api_server \\")
print(f"    --model {MERGED_DIR} \\")
print(f"    --served-model-name bastion-32b \\")
print(f"    --tensor-parallel-size 4 \\")
print(f"    --max-model-len 8192 \\")
print(f"    --gpu-memory-utilization 0.85 \\")
print(f"    --max-num-seqs 64 \\")
print(f"    --host 0.0.0.0 --port 8000 > /workspace/vllm_v6.log 2>&1 &")
