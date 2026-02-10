# 🚀 BASTION RISK INTELLIGENCE TRAINING - START NOW

**Everything is ready. Start training immediately.**

---

## CLUSTER STATUS: ✅ READY

| Component | Status | Location |
|-----------|--------|----------|
| SSH | `ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178` | - |
| Python | `/opt/sys-venv/bin/python3` | - |
| PyTorch | 2.10.0+cu128 (RTX 5090 compatible) | - |
| GPUs | 4x RTX 5090 (32GB each, 128GB total) | - |
| Merged Model | ✅ Ready | `/root/models/iros-merged` |
| Training Data | ✅ Ready | `/root/training/` |
| Training Scripts | ✅ Ready | `/root/scripts/` |

---

## 📚 TRAINING CORPUS: 797 EXAMPLES

### Location: `/root/training/`

### IROS Analyst Files (755 examples):
| File | Examples | Content |
|------|----------|---------|
| `iros_72b_train.jsonl` | 600 | Core market analysis, trade ideas, MCF scoring |
| `iros_72b_eval.jsonl` | 67 | Evaluation set |
| `mcf_intelligence_reports.jsonl` | 39 | Full MCF analysis reports |
| `bastion_api_knowledge.jsonl` | 9 | Helsinki, Coinglass, Whale Alert API knowledge |
| `trade_rejection_examples.jsonl` | 7 | Rejecting bad trade setups |
| `enhanced_tier1_tda.jsonl` | 6 | Topological Data Analysis |
| `adversarial_examples.jsonl` | 6 | Edge cases, manipulation handling |
| `combined_analysis.jsonl` | 6 | Multi-source analysis combining all APIs |
| `enhanced_tier1_rl.jsonl` | 5 | Reinforcement learning concepts |
| `helsinki_endpoints.jsonl` | 4 | Helsinki VM endpoint deep dives |
| `chain_of_thought_examples.jsonl` | 3 | Step-by-step reasoning |
| `mcf_scoring_examples.jsonl` | 3 | MCF scoring walkthroughs |

### Risk Intelligence Files (42 examples):
| File | Examples | Content |
|------|----------|---------|
| `risk_intelligence_train.jsonl` | 17 | Core position management |
| `risk_intelligence_expanded.jsonl` | 20 | Expanded scenarios (ETH, SOL, AVAX, LINK, etc.) |
| `risk_intelligence_eval.jsonl` | 5 | Evaluation set |

---

## 🎯 RISK INTELLIGENCE TRAINING SCENARIOS

The Risk Intelligence model learns to:

### Exit Priorities (MCF Logic):
1. **Hard Stop** - If breached, exit immediately
2. **Safety Net** - Secondary protection level
3. **Guarding Line** - Dynamic S/R breaks
4. **Take Profits** - TP1/TP2/TP3 with scaling
5. **Trailing Stop** - Lock in gains
6. **Time Exit** - Stagnant trade cleanup

### Scenarios Covered:
- Partial take profits (25%, 33%, 50%, 75%)
- Short positions with squeeze risk
- Underwater positions with bullish divergence
- Stop breaches requiring immediate exit
- Extreme funding warnings (0.08%, 0.12%)
- Whale deposit alerts
- Options max pain influence
- Volume profile support/resistance
- Smart money vs retail divergence
- Liquidation cluster stop optimization
- Time-in-trade thesis degradation
- Multi-position portfolio management
- Flash crash emergency exits
- Re-entry evaluations
- Hold decisions (when to do nothing)

### Output Format:
```json
{
  "action": "PARTIAL_TP",
  "quantity_pct": 33,
  "reasoning": "Position at +1.3R with TP1 approaching...",
  "adjustments": [{"type": "TRAIL_STOP", "new_stop": 3200}],
  "confidence": 82
}
```

---

## 📡 DATA SOURCES AI KNOWS

### Helsinki VM (FREE - 33 endpoints)
- **Base:** `http://77.42.29.188:5002`
- CVD, orderbook imbalance, volatility regime
- Smart money signals, liquidation estimates
- Key: `/quant/full/{symbol}` returns everything

### Coinglass Premium ($299/mo)
- **Base:** `https://open-api-v3.coinglass.com/api`
- Liquidation heatmaps, funding rates
- Open interest, Long/Short ratios
- Top trader positioning

### Whale Alert Premium ($29.95/mo)
- **Base:** `https://api.whale-alert.io/v1`
- On-chain whale tracking
- Exchange flows (deposits/withdrawals)
- Stablecoin mints/burns

---

## 🔧 TRAINING CONFIGURATION

### Recommended Settings:
```python
BASE_MODEL = "/root/models/iros-merged"  # The merged H200 LoRA model
LORA_R = 16
LORA_ALPHA = 32
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3-5
BATCH_SIZE = 2  # Per GPU
GRADIENT_ACCUMULATION = 8
MAX_SEQ_LENGTH = 2048
```

### System Prompt for Risk Intelligence:
```
You are BASTION Risk Intelligence - an autonomous trade management AI. 
You monitor live positions and make execution decisions. 
Output JSON with action, reasoning, and confidence. 
PRIORITY ORDER: 1) Hard Stop 2) Safety Net Break 3) Guarding Line Break 4) Targets 5) Trailing Stop 6) Time Exit. 
Core philosophy: Exit on STRUCTURE BREAKS, not arbitrary targets. Let winners run when structure holds.
```

---

## 🚀 START TRAINING

### Step 1: Combine Risk Intelligence training files
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "cat /root/training/risk_intelligence_train.jsonl /root/training/risk_intelligence_expanded.jsonl > /root/training/risk_combined.jsonl && wc -l /root/training/risk_combined.jsonl"
```
Should output: 37 lines (37 training examples)

### Step 2: Create training script on cluster
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "cat > /root/train_risk_lora.py << 'SCRIPT'
import os
os.environ['HF_HOME'] = '/workspace/.hf_home'

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Config
BASE_MODEL = '/root/models/iros-merged'
TRAIN_DATA = '/root/training/risk_combined.jsonl'
EVAL_DATA = '/root/training/risk_intelligence_eval.jsonl'
OUTPUT_DIR = '/root/checkpoints/risk-intelligence-lora'

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)

print('Configuring LoRA...')
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print('Loading datasets...')
train_dataset = load_dataset('json', data_files=TRAIN_DATA, split='train')
eval_dataset = load_dataset('json', data_files=EVAL_DATA, split='train')

def format_chat(example):
    messages = example['messages']
    text = ''
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == 'system':
            text += f'<|im_start|>system\n{content}<|im_end|>\n'
        elif role == 'user':
            text += f'<|im_start|>user\n{content}<|im_end|>\n'
        elif role == 'assistant':
            text += f'<|im_start|>assistant\n{content}<|im_end|>\n'
    return {'text': text}

train_dataset = train_dataset.map(format_chat)
eval_dataset = eval_dataset.map(format_chat)

print('Setting up trainer...')
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    warmup_steps=10,
    logging_steps=1,
    save_steps=50,
    eval_strategy='steps',
    eval_steps=25,
    fp16=True,
    report_to='none',
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field='text',
    max_seq_length=2048,
    tokenizer=tokenizer,
)

print('Starting training...')
trainer.train()

print('Saving model...')
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print('=== TRAINING COMPLETE ===')
print(f'Model saved to: {OUTPUT_DIR}')
SCRIPT"
```

### Step 3: Install trl if needed
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "/opt/sys-venv/bin/pip install trl"
```

### Step 4: Start training with nohup
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "nohup /opt/sys-venv/bin/python3 /root/train_risk_lora.py > /root/training_output.log 2>&1 & echo 'Training PID:' \$!"
```

### Step 5: Monitor training
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "tail -f /root/training_output.log"
```

---

## 📊 EXPECTED TRAINING TIME

- 37 training examples
- 5 epochs
- ~10-15 minutes on 4x RTX 5090

---

## 🎯 AFTER TRAINING

The Risk Intelligence LoRA will be saved at:
`/root/checkpoints/risk-intelligence-lora/`

### Test the trained model:
```python
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained('/root/models/iros-merged', ...)
model = PeftModel.from_pretrained(base, '/root/checkpoints/risk-intelligence-lora')

# Test with a position state
prompt = """POSITION: BTC LONG @ $94,500
CURRENT: $96,200 (+1.8%)
STOP: $93,800
TP1: $97,000 (untouched)
R-MULTIPLE: +1.2R
FUNDING: -0.02%
CVD: bullish divergence

DECISION?"""
```

---

## 🏆 END GOAL

Two models on the cluster:
1. **IROS Analyst** (GPU 0-1): `/root/models/iros-merged` - Market analysis
2. **Risk Intelligence** (GPU 2-3): Merged + Risk LoRA - Trade management

Deploy with vLLM for production inference.

---

**LET'S GO! START TRAINING NOW!** 🚀

