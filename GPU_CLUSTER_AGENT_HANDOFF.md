# 🚀 BASTION GPU CLUSTER AGENT HANDOFF

**Copy-paste this entire document to a new Cursor agent if the previous one dies.**

---

## 📋 PROJECT OVERVIEW

You are helping set up and train BASTION AI models on a **4x RTX 5090 GPU cluster** (or similar).

### What We're Building:
1. **IROS Analyst** - Market analysis AI (H200 LoRA already trained)
2. **Risk Intelligence** - Autonomous trade management AI (needs training)

### Current State:
- ✅ H200 LoRA adapters exist (trained on MCF methodology, Helsinki endpoints)
- ✅ Training corpus ready (797 examples in JSONL format)
- ✅ Risk Intelligence training data CREATED (37 examples)
- ✅ API knowledge training data CREATED (15 examples)
- ✅ Training scripts ready (4-GPU parallel)
- ❌ Need to: Load H200 LoRA onto 5090 cluster
- ❌ Need to: Fine-tune Risk Intelligence model

---

## 📁 FILE LOCATIONS (On User's Windows Machine)

```
C:\Users\Banke\IROS_72B_TRAINING_CORPUS\
├── FINAL_TRAINING\
│   ├── iros_72b_train.jsonl               # 600 training examples
│   ├── iros_72b_eval.jsonl                # 67 eval examples
│   ├── helsinki_endpoints.jsonl           # 4 - Helsinki API training
│   ├── mcf_scoring_examples.jsonl         # 3 - MCF methodology
│   ├── adversarial_examples.jsonl         # 6 - Edge cases
│   ├── chain_of_thought_examples.jsonl    # 3 - Reasoning
│   ├── enhanced_tier1_rl.jsonl            # 5 - RL concepts
│   ├── enhanced_tier1_tda.jsonl           # 6 - TDA concepts
│   ├── trade_rejection_examples.jsonl     # 7 - Bad trade rejection
│   ├── mcf_intelligence_reports.jsonl     # 39 - Full analysis reports
│   │
│   ├── bastion_api_knowledge.jsonl        # 9 - API endpoint training ⭐ NEW
│   ├── combined_analysis.jsonl            # 6 - Multi-source analysis ⭐ NEW
│   │
│   ├── risk_intelligence_train.jsonl      # 17 - Risk management training ⭐
│   ├── risk_intelligence_expanded.jsonl   # 20 - Expanded scenarios ⭐ NEW
│   ├── risk_intelligence_eval.jsonl       # 5 - Risk eval
│   │
│   └── training_manifest.json             # Metadata (797 total examples)
│
├── scripts\
│   ├── train_4gpu_parallel.py             # Main training script
│   ├── train_iros_lora.py                 # LoRA training
│   └── train_with_deepspeed.py            # DeepSpeed version
│
├── MCF_RISK_EXIT_LOGIC_EXTRACTION.md      # Full MCF risk logic documentation
│
└── rag_database\                          # ChromaDB vectors
```

---

## 📊 TRAINING DATA SUMMARY

### IROS Analyst Model (688 train + 67 eval = 755)
| File | Examples | Content |
|------|----------|---------|
| iros_72b_train.jsonl | 600 | Core market analysis training |
| mcf_intelligence_reports.jsonl | 39 | Full MCF analysis reports |
| bastion_api_knowledge.jsonl | 9 | API endpoints understanding |
| trade_rejection_examples.jsonl | 7 | Bad trade rejection |
| enhanced_tier1_tda.jsonl | 6 | TDA concepts |
| adversarial_examples.jsonl | 6 | Edge cases |
| combined_analysis.jsonl | 6 | Multi-source analysis |
| enhanced_tier1_rl.jsonl | 5 | RL concepts |
| helsinki_endpoints.jsonl | 4 | Helsinki API deep dive |
| chain_of_thought_examples.jsonl | 3 | Reasoning chains |
| mcf_scoring_examples.jsonl | 3 | MCF scoring |
| iros_72b_eval.jsonl | 67 | Evaluation set |

### Risk Intelligence Model (37 train + 5 eval = 42)
| File | Examples | Content |
|------|----------|---------|
| risk_intelligence_train.jsonl | 17 | Core risk management |
| risk_intelligence_expanded.jsonl | 20 | Expanded scenarios |
| risk_intelligence_eval.jsonl | 5 | Evaluation set |

---

## 📡 DATA SOURCES THE AI KNOWS

The training data teaches the AI about these paid data sources:

### Helsinki VM (FREE - 33 endpoints)
- **Base URL:** http://77.42.29.188:5002
- **Capabilities:** CVD, orderbook, volatility regime, smart money, liquidations
- **Key endpoint:** `/quant/full/{symbol}` - returns everything

### Coinglass Premium ($299/month)
- **Base URL:** https://open-api-v3.coinglass.com/api
- **Capabilities:** Liquidation heatmaps, funding rates, OI, L/S ratios, options
- **Key insight:** Funding extremes, top trader positioning

### Whale Alert Premium ($29.95/month)
- **Base URL:** https://api.whale-alert.io/v1
- **Capabilities:** On-chain tracking, exchange flows, whale transactions
- **Key insight:** Exchange deposits/withdrawals, stablecoin mints

---

## 🎯 YOUR MISSION

### Phase 1: Setup GPU Cluster

1. **Rent 4x 5090 cluster** (Vast.ai, RunPod, or Lambda)
   - Need ~128GB total VRAM
   - Ubuntu 22.04 with CUDA 12.x
   - At least 200GB disk space

2. **SSH into cluster and setup environment:**
```bash
# Install dependencies
pip install torch transformers accelerate peft datasets bitsandbytes
pip install flash-attn --no-build-isolation
pip install vllm  # For inference later

# Create directories
mkdir -p /root/training /root/models /root/checkpoints
```

3. **Transfer files from Windows to cluster:**
```bash
# On Windows (PowerShell), transfer ALL training data:
scp -r "C:\Users\Banke\IROS_72B_TRAINING_CORPUS\FINAL_TRAINING\*" root@<CLUSTER_IP>:/root/training/

# Transfer training scripts:
scp -r "C:\Users\Banke\IROS_72B_TRAINING_CORPUS\scripts\*" root@<CLUSTER_IP>:/root/scripts/

# Transfer MCF documentation:
scp "C:\Users\Banke\IROS_72B_TRAINING_CORPUS\MCF_RISK_EXIT_LOGIC_EXTRACTION.md" root@<CLUSTER_IP>:/root/training/
```

### Phase 2: Load H200 LoRA Adapters

**If H200 LoRA checkpoint exists, transfer it:**
```bash
# Find where H200 LoRA was saved and transfer to cluster
scp -r /path/to/h200-iros-lora root@<CLUSTER_IP>:/root/models/h200-iros-lora/
```

**Test the LoRA loads correctly:**
```python
# test_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("Loading H200 LoRA...")
model = PeftModel.from_pretrained(model, "/root/models/h200-iros-lora")

print("Testing inference...")
inputs = tokenizer("What is the MCF scoring methodology?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

### Phase 3: Merge H200 LoRA (Recommended)

```python
# merge_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
LORA_PATH = "/root/models/h200-iros-lora"
MERGED_OUTPUT = "/root/models/iros-base-merged"

print("Loading base + LoRA...")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="cpu")
model = PeftModel.from_pretrained(model, LORA_PATH)

print("Merging LoRA into base weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_OUTPUT}...")
model.save_pretrained(MERGED_OUTPUT)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(MERGED_OUTPUT)

print("Done! Merged model ready for Risk Intelligence fine-tuning.")
```

### Phase 4: Train Risk Intelligence LoRA

**Training data READY (37 examples + 5 eval):**
- `risk_intelligence_train.jsonl` - 17 core scenarios
- `risk_intelligence_expanded.jsonl` - 20 expanded scenarios
- `risk_intelligence_eval.jsonl` - 5 evaluation examples

**Combine training files first:**
```bash
cat /root/training/risk_intelligence_train.jsonl /root/training/risk_intelligence_expanded.jsonl > /root/training/risk_intelligence_combined.jsonl
```

**Training command:**
```bash
cd /root/scripts

# Train Risk Intelligence LoRA on merged IROS base
accelerate launch --num_processes 4 train_4gpu_parallel.py \
    --base_model /root/models/iros-base-merged \
    --train_data /root/training/risk_intelligence_combined.jsonl \
    --eval_data /root/training/risk_intelligence_eval.jsonl \
    --output_dir /root/checkpoints/risk-intelligence-lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_epochs 5 \
    --learning_rate 2e-4
```

---

## 🔧 SYSTEM PROMPTS

### Risk Intelligence:
```
You are BASTION Risk Intelligence - an autonomous trade management AI. 
You monitor live positions and make execution decisions. 
Output JSON with action, reasoning, and confidence. 
PRIORITY ORDER: 1) Hard Stop 2) Safety Net Break 3) Guarding Line Break 4) Targets 5) Trailing Stop 6) Time Exit. 
Core philosophy: Exit on STRUCTURE BREAKS, not arbitrary targets. Let winners run when structure holds.
```

### IROS Analyst:
```
You are BASTION - an institutional-grade crypto trading AI with access to premium data sources including Helsinki VM (33 free endpoints), Coinglass Premium ($299/mo for liquidations, OI, funding, L/S ratios), and Whale Alert Premium ($29.95/mo for on-chain tracking). Provide comprehensive market analysis by combining multiple data sources.
```

---

## 🔧 TRAINING CONFIGURATION

```python
# For Risk Intelligence LoRA
BASE_MODEL = "/root/models/iros-base-merged"  # Or Qwen2.5-32B if no H200 LoRA
LORA_R = 16
LORA_ALPHA = 32
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
BATCH_SIZE = 2  # Per GPU
GRADIENT_ACCUMULATION = 8
MAX_SEQ_LENGTH = 2048  # Longer for position context
```

---

## 🎯 END GOAL

Two models running on the 4x 5090 cluster:

```
GPU 0-1: IROS Analyst (merged H200 LoRA)
         - Handles: "Should I long BTC?" queries
         - Market analysis, trade ideas
         - Knows all API endpoints (Helsinki, Coinglass, Whale Alert)

GPU 2-3: Risk Intelligence (H200 merged + Risk LoRA)
         - Handles: Live position management
         - Outputs: {"action": "TP_50%", "price": 98500}
         - Connected to order execution layer
```

---

## ⚠️ IF AGENT DIES - CHECKPOINT STATUS

Update this section before agent dies:

```
LAST COMPLETED STEP: Files transferred to cluster
CURRENT WORKING ON: pip install torch (agent died during download)
CLUSTER IP: 74.48.140.178
CLUSTER SSH PORT: 26796
SSH COMMAND: ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178
FILES TRANSFERRED: YES - to /root/training/ and /root/scripts/
H200 LORA TRANSFERRED: YES - to /root/models/h200-lora/iros-mega-lora-2026-02-01/
H200 LORA SOURCE: C:\Users\Banke\IROS_Lives\whale-terminal\iros-mega-lora-2026-02-01\
H200 LORA CONFIG: r=128, alpha=256, 93.7% accuracy
PIP INSTALL: IN PROGRESS (may have completed)
TORCH VERIFIED: NO - need to check
LORA LOADED: NO
MERGED MODEL CREATED: NO
RISK TRAINING DATA: COMPLETE (37 train + 5 eval)
API KNOWLEDGE DATA: COMPLETE (15 examples)
TRAINING STARTED: NO
TRAINING EPOCH: 0/3
```

### RESUME STEPS:
1. Check pip install status: `pip list | grep -E 'torch|transformers|peft'`
2. If not done: Re-run pip install
3. Verify CUDA: `python -c 'import torch; print(torch.cuda.is_available())'`
4. Transfer new training files (bastion_api_knowledge.jsonl, risk_intelligence_expanded.jsonl, combined_analysis.jsonl)
5. Test LoRA loading
6. Merge H200 LoRA
7. Train Risk Intelligence

---

## 📞 CONTEXT FOR NEW AGENT

If you're a new agent picking this up:

1. Read this entire document first
2. Ask user: "What was the last completed step?"
3. Check the checkpoint status above
4. Continue from where the previous agent left off
5. Update checkpoint status as you complete each step

**Key contacts:**
- Helsinki VM: http://77.42.29.188:5002 (free, no auth)
- Training corpus: C:\Users\Banke\IROS_72B_TRAINING_CORPUS\
- BASTION codebase: C:\Users\Banke\BASTION\BASTION\

---

## 🔑 IMPORTANT COMMANDS REFERENCE

```bash
# Check GPU status
nvidia-smi

# Check VRAM usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Kill stuck processes
pkill -f python

# Monitor training
tail -f /root/checkpoints/*/trainer_log.txt

# Test model inference
python -c "from transformers import pipeline; p = pipeline('text-generation', model='/root/models/iros-base-merged', device_map='auto'); print(p('Hello BASTION'))"

# Count training examples
wc -l /root/training/*.jsonl
```

---

**This document lives at: `C:\Users\Banke\BASTION\BASTION\GPU_CLUSTER_AGENT_HANDOFF.md`**

**Last updated: 2026-02-09**

**Total training examples: 797 (725 train + 72 eval)**
