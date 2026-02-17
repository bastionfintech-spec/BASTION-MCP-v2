#!/usr/bin/env python3
"""
BASTION Risk Intelligence v2 - Fine-Tuning Script
Trains on top of bastion-risk-merged with standardized action format.
Uses QLoRA (4-bit quantized base + LoRA adapters) on 4x RTX 5090 GPUs.

Key improvements over v1:
- Standardized 9 action types matching execution engine
- Clean execution.exit_pct (integer) and execution.stop_price (float) fields
- AI reasons about WHY specific exit percentages
- ~200 diverse training examples
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

# ============================================================
# CONFIG
# ============================================================
# Train on top of the EXISTING merged model (keeps IROS + Risk v1 intelligence)
MODEL_PATH = "/root/models/bastion-risk-merged"
DATA_PATH = "/workspace/bastion_risk_v2_standardized.jsonl"
OUTPUT_DIR = "/workspace/bastion-risk-v2-lora"
MERGED_OUTPUT = "/root/models/bastion-risk-v2-merged"

# LoRA hyperparameters (same as v1 for consistency)
LORA_R = 32              # Rank
LORA_ALPHA = 64          # Scaling factor (2x rank)
LORA_DROPOUT = 0.05      # Regularization
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training hyperparameters
NUM_EPOCHS = 3            # ~200 examples * 3 = ~600 steps; less epochs since retraining on merged
BATCH_SIZE = 1            # Per-device
GRADIENT_ACCUM = 4        # Effective batch = 1 * 4 = 4 (slightly smaller for more updates)
LEARNING_RATE = 1.5e-5    # Slightly lower LR since we're fine-tuning already fine-tuned model
MAX_SEQ_LEN = 4096        # Max sequence length
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 5
SAVE_STEPS = 50

print("=" * 60)
print("BASTION Risk Intelligence v2 - Fine-Tuning")
print("=" * 60)
print(f"Model: {MODEL_PATH}")
print(f"Data: {DATA_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} - {free/(1024**3):.1f}GB free / {total/(1024**3):.1f}GB total")

# ============================================================
# LOAD TOKENIZER
# ============================================================
print("\n[1/7] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"
print(f"  Pad token: {tokenizer.pad_token}")
print(f"  EOS token: {tokenizer.eos_token}")

# ============================================================
# LOAD AND FORMAT DATA
# ============================================================
print("\n[2/7] Loading training data...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line.strip()) for line in f if line.strip()]

print(f"  Loaded {len(raw_data)} examples")

# Validate data format
action_counts = {}
for ex in raw_data:
    msgs = ex.get("messages", [])
    if len(msgs) >= 3:
        try:
            assistant_msg = msgs[2]["content"]
            parsed = json.loads(assistant_msg)
            action = parsed.get("action", "UNKNOWN")
            action_counts[action] = action_counts.get(action, 0) + 1
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

print(f"  Action distribution:")
for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
    print(f"    {action}: {count}")

# Format using chat template
def format_example(example):
    """Apply ChatML template to messages."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_example, remove_columns=["messages"])
print(f"  Formatted {len(dataset)} examples")

# Check token lengths
sample_tokens = tokenizer(dataset[0]["text"], return_tensors="pt")
print(f"  Sample token length: {sample_tokens['input_ids'].shape[1]}")

# ============================================================
# LOAD MODEL WITH QUANTIZATION
# ============================================================
print("\n[3/7] Loading model with 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.config.use_cache = False
print(f"  Model loaded. Parameters: {model.num_parameters():,}")
print(f"  Device map: {set(model.hf_device_map.values())}")

# ============================================================
# APPLY LoRA
# ============================================================
print("\n[4/7] Applying LoRA configuration...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(f"  Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# ============================================================
# TRAINING
# ============================================================
print("\n[5/7] Starting training...")
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    packing=False,
    report_to="none",
    seed=42,
    dataloader_pin_memory=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
)

# Train
train_result = trainer.train()
print(f"\n  Training complete!")
print(f"  Total steps: {trainer.state.global_step}")
print(f"  Final loss: {train_result.training_loss:.4f}")

# Save LoRA adapter
print("\n[6/7] Saving LoRA adapter...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"  Adapter saved to {OUTPUT_DIR}")

# ============================================================
# MERGE LoRA INTO BASE MODEL (ON CPU)
# ============================================================
print("\n[7/7] Merging LoRA adapter into base model (CPU)...")
print("  This takes ~5-7 minutes with 1TB RAM. Using CPU to avoid quality loss from 4-bit merge.")

# Clear GPU memory
del model
del trainer
torch.cuda.empty_cache()
import gc
gc.collect()

# Load base model on CPU for clean fp16 merge
print("  Loading base model on CPU (fp16)...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cpu",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

# Load and merge LoRA
from peft import PeftModel
print("  Loading LoRA adapter...")
merged_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR, device_map="cpu")
print("  Merging weights...")
merged_model = merged_model.merge_and_unload()

# Save merged model
print(f"  Saving merged model to {MERGED_OUTPUT}...")
merged_model.save_pretrained(MERGED_OUTPUT, safe_serialization=True)
tokenizer.save_pretrained(MERGED_OUTPUT)

print("\n" + "=" * 60)
print("FINE-TUNING v2 COMPLETE!")
print("=" * 60)
print(f"LoRA adapter: {OUTPUT_DIR}")
print(f"Merged model: {MERGED_OUTPUT}")
print(f"Training loss: {train_result.training_loss:.4f}")
print(f"\nTo serve with vLLM:")
print(f"  # First stop the current vLLM instance")
print(f"  # Then launch with the new model:")
print(f"  export VLLM_USE_V1=0")
print(f"  /opt/sys-venv/bin/python3 -m vllm.entrypoints.openai.api_server \\")
print(f"    --model {MERGED_OUTPUT} \\")
print(f"    --served-model-name bastion-32b \\")
print(f"    --tensor-parallel-size 4 \\")
print(f"    --enforce-eager \\")
print(f"    --dtype float16 \\")
print(f"    --max-model-len 8192 \\")
print(f"    --host 0.0.0.0 \\")
print(f"    --port 8000 \\")
print(f"    --gpu-memory-utilization 0.90")
