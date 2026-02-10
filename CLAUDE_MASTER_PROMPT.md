# 🏰 BASTION MASTER CONTEXT - CLAUDE ULTRA DEPLOYMENT GUIDE

**Last Updated:** February 10, 2026  
**User Account:** bastionfintech@gmail.com  
**Project Status:** GPU Cluster Active, vLLM Deployment in Progress

---

## ⚠️ CRITICAL CONTEXT: WHY YOU'RE HERE

The user (Banke) has spent **all day** fighting with Cursor agents that keep crashing/dying during long-running tasks. They have a **4x RTX 5090 GPU cluster on Vast.ai** that costs **~$1.91/hour** and they need to:

1. **Deploy vLLM** to serve their fine-tuned AI model
2. **Connect BASTION terminal** to the AI for live trading analysis
3. **Enable Risk Intelligence** for autonomous position management

**YOU MUST EXECUTE COMMANDS YOURSELF using Desktop Commander.** Do NOT just tell the user what to do - USE THE TOOLS.

---

## 🖥️ GPU CLUSTER CONNECTION (ACTIVE NOW)

```
SSH Command: ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178
IP Address: 74.48.140.178
SSH Port: 26796
Hardware: 4x NVIDIA RTX 5090 (32GB VRAM each = 128GB total)
Storage: 1TB NVMe
Python: /opt/sys-venv/bin/python3
Cost: ~$1.91/hour (cluster is actively running and billing)
```

### GPU Status (Verified Working):
```
NVIDIA GeForce RTX 5090, 2 MiB used, 32607 MiB total (x4)
```

All 4 GPUs are **idle with minimal memory usage** - ready for vLLM deployment.

---

## 📁 WHAT EXISTS ON THE CLUSTER

### Models & Adapters:
```
/root/models/iros-merged/           # ~65GB merged Qwen2.5-32B with IROS LoRA
/root/checkpoints/risk-intelligence-lora/  # Trained Risk Intelligence LoRA
/workspace/bastion-mega-lora/       # Backup LoRA location (check if exists)
```

### Training Data (Already Transferred):
```
/root/training/
├── iros_72b_train.jsonl            # 600 examples
├── iros_72b_eval.jsonl             # 67 examples
├── risk_intelligence_train.jsonl   # 17 examples
├── risk_intelligence_expanded.jsonl # 20 examples
├── risk_intelligence_eval.jsonl    # 5 examples
├── bastion_api_knowledge.jsonl     # 9 examples
├── combined_analysis.jsonl         # 6 examples
├── mcf_intelligence_reports.jsonl  # 39 examples
└── ... (797 total training examples)
```

---

## 🚀 IMMEDIATE TASK: DEPLOY vLLM

### The Problem We've Been Facing:
- vLLM 0.15.x has a **broken v1 engine** that crashes with "Engine core initialization failed"
- Need to either downgrade to **vLLM 0.6.3** or use **--enforce-eager** flag
- The `VLLM_USE_V1=0` environment variable must be set

### EXECUTE THESE COMMANDS IN ORDER:

#### Step 1: Kill any existing processes
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "pkill -9 -f vllm; pkill -9 -f python; echo KILLED"
```

#### Step 2: Check current vLLM version
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "/opt/sys-venv/bin/pip show vllm | grep Version"
```

#### Step 3: If version is 0.15.x, downgrade to 0.6.3
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "/opt/sys-venv/bin/pip install vllm==0.6.3 --force-reinstall"
```
**⚠️ THIS TAKES 2-5 MINUTES - DO NOT INTERRUPT**

#### Step 4: Start vLLM with working configuration
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "VLLM_USE_V1=0 nohup /opt/sys-venv/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /root/models/iros-merged \
    --tensor-parallel-size 4 \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 4096 \
    --trust-remote-code \
    --enforce-eager \
    > /root/vllm_server.log 2>&1 &"
```

#### Step 5: Wait and check logs
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "sleep 60 && tail -100 /root/vllm_server.log"
```

**Look for:** "Uvicorn running on http://0.0.0.0:8000"

#### Step 6: Test the API
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "curl -s http://localhost:8000/health"
```

```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "curl -s http://localhost:8000/v1/models"
```

#### Step 7: Test inference
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{\"model\": \"/root/models/iros-merged\", \"messages\": [{\"role\": \"user\", \"content\": \"What is MCF scoring?\"}], \"max_tokens\": 100}'"
```

---

## 🏗️ WHAT BASTION IS

BASTION is an **institutional-grade crypto trading terminal** with:

### Frontend (generated-page.html)
- Live price charts (LightweightCharts)
- Position management panel
- Risk simulation with Monte Carlo
- Order flow analysis (CVD, OI changes)
- On-chain whale tracking
- MCF Labs reports
- Collapsible panels with layout saving

### Backend (api/terminal_api.py - FastAPI)
- Deployed on Railway at production URL
- Connects to Helsinki VM, Coinglass, Whale Alert
- User authentication via Supabase
- Exchange integration (Bitunix, etc.)
- Live alerts to Telegram

### AI Components (What We're Deploying)
1. **IROS Analyst** - Market analysis, trade ideas, data interpretation
2. **Risk Intelligence** - Autonomous position management, executes TP/SL/trailing stops

---

## 📊 DATA SOURCES (AI KNOWS THESE)

### 1. Helsinki VM (FREE - 33 endpoints)
```
Base URL: http://77.42.29.188:5002
No auth required, unlimited rate limit

Key Endpoints:
- /quant/full/{symbol}     → All data for a symbol
- /quant/cvd/{symbol}      → Cumulative Volume Delta
- /quant/liquidation-estimate/{symbol} → Liquidation clusters
- /quant/options-iv/{symbol} → Real-time price + IV
- /quant/smart-money/{symbol} → Institutional flow
- /quant/volatility/{symbol} → Regime detection
- /sentiment/fear-greed    → Market sentiment
- /quant/dominance         → BTC/ETH dominance
```

### 2. Coinglass Premium ($299/month)
```
Base URL: https://open-api-v3.coinglass.com/api
Auth: CG-API-KEY header
API Key: 03e5a43afaa4489384cb935b9b2ea16b

Key Endpoints:
- /futures/liquidation/heatmap      → Liquidation clusters
- /futures/liquidation/aggregated-history → Historical liqs
- /futures/openInterest/exchange-list → OI by exchange
- /futures/fundingRate/exchange-list → Funding rates
- /futures/topLongShortAccountRatio/history → Whale L/S ratio
- /options/info → Put/call ratio, max pain
```

### 3. Whale Alert Premium ($29.95/month)
```
Base URL: https://api.whale-alert.io/v1
Auth: api_key query parameter
API Key: OsCbgowziN4kMRnC6WhgokKXW3bfVKcQ

Key Endpoints:
- /transactions → Recent whale txs
- /transaction/{blockchain}/{hash} → Specific tx

WebSocket: wss://ws.whale-alert.io (real-time streaming)
```

---

## 🧠 AI MODEL DETAILS

### Base Model:
- **Qwen/Qwen2.5-Coder-32B-Instruct** (or AWQ quantized variant)
- 32 billion parameters
- Split across 4 GPUs via tensor parallelism (~8B params per GPU)

### Trained LoRA Adapters:
1. **IROS Mega LoRA** (from H200 training)
   - Config: r=128, alpha=256
   - Accuracy: 93.7%
   - Trained on: MCF methodology, Helsinki endpoints, market analysis
   - Location: Merged into `/root/models/iros-merged`

2. **Risk Intelligence LoRA**
   - Trained on: 37 position management scenarios
   - Outputs: JSON actions `{"action": "TP_50%", "price": 98500, "confidence": 0.92}`
   - Location: `/root/checkpoints/risk-intelligence-lora`

### System Prompts:

**For IROS Analyst (chat/analysis):**
```
You are BASTION - an institutional-grade crypto trading AI with access to Helsinki VM (33 endpoints), Coinglass Premium (liquidations, funding, OI), and Whale Alert (on-chain tracking). Provide comprehensive market analysis combining multiple data sources.
```

**For Risk Intelligence (position management):**
```
You are BASTION Risk Intelligence - an autonomous trade management AI. You monitor live positions and make execution decisions. Output JSON with action, reasoning, and confidence. 
PRIORITY ORDER: 1) Hard Stop 2) Safety Net 3) Guarding Line 4) Targets 5) Trail 6) Time Exit.
Core philosophy: Exit on STRUCTURE BREAKS, not arbitrary targets. Let winners run when structure holds.
```

---

## 🔧 MCF (Multi-Confirmation Framework) - TRADING METHODOLOGY

The AI is trained on MCF, which includes:

### Risk Exit Logic Priority:
1. **Hard Stop** - Maximum loss threshold, non-negotiable
2. **Safety Net Break** - Secondary structure violation
3. **Guarding Line Break** - Trailing structure level
4. **Take Profit Targets** - T1, T2, T3 based on structure
5. **Trailing Stop** - ATR-based dynamic stop
6. **Time Exit** - Position duration limits

### MCF Scoring:
- Analyzes multiple timeframes (5m, 15m, 1H, 4H)
- Combines order flow, liquidations, whale activity
- Generates institutional-grade reports

---

## 📡 HOW THE 4 GPUs WORK TOGETHER

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM SERVER (Port 8000)                  │
│                  Tensor Parallel Mode (TP=4)                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   The 32B model is SPLIT across all 4 GPUs:                 │
│                                                             │
│   GPU 0: Layers 0-15   (~16GB VRAM)                         │
│   GPU 1: Layers 16-31  (~16GB VRAM)                         │
│   GPU 2: Layers 32-47  (~16GB VRAM)                         │
│   GPU 3: Layers 48-63  (~16GB VRAM)                         │
│                                                             │
│   ALL 4 GPUs work on EVERY request together                 │
│   = Faster inference than single GPU                        │
│   = Can handle 10-20 concurrent users                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘

API Endpoint: http://74.48.140.178:8000
```

### OpenAI-Compatible Endpoints:
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | Chat with BASTION |
| `/v1/completions` | POST | Raw text completion |
| `/v1/models` | GET | List loaded models |
| `/health` | GET | Health check |

---

## 🔌 CONNECTING BASTION TO vLLM

Once vLLM is running, update the BASTION backend:

### In `api/terminal_api.py`:
```python
import httpx

VLLM_URL = "http://74.48.140.178:8000/v1/chat/completions"

async def ask_bastion(user_message: str, system_prompt: str = None):
    """Query the BASTION AI model."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(VLLM_URL, json={
            "model": "/root/models/iros-merged",
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        })
        return response.json()["choices"][0]["message"]["content"]

# For Risk Intelligence (autonomous trading)
async def evaluate_position(position_data: dict) -> dict:
    """AI evaluates position and returns action."""
    system = """You are BASTION Risk Intelligence - an autonomous trade management AI. 
Output JSON with action, reasoning, and confidence.
PRIORITY ORDER: 1) Hard Stop 2) Safety Net 3) Guarding Line 4) Targets 5) Trail 6) Time Exit."""
    
    user_msg = f"Evaluate this position and recommend action:\n{json.dumps(position_data)}"
    
    response = await ask_bastion(user_msg, system)
    return json.loads(response)  # {"action": "HOLD", "reasoning": "...", "confidence": 0.85}
```

---

## 📱 TELEGRAM ALERTS

BASTION sends live alerts to Telegram:
- Channel: @bastion_signals (or configured channel)
- Bot Token: Configured in environment
- Alert types: Liquidation warnings, whale movements, momentum shifts

---

## 📊 GLOBAL STATISTICS (Marketing)

BASTION tracks cumulative stats for the landing page:

```python
bastion_stats = {
    "total_users": 0,                    # Increments on registration
    "total_exchanges_connected": 0,      # Increments on exchange connect
    "total_positions_analyzed": 0,       # Increments on position analysis
    "total_portfolio_managed_usd": 0.0   # CUMULATIVE - adds on every connect
}
```

**Important:** `total_portfolio_managed_usd` is CUMULATIVE - if a user connects with $7k, disconnects, and reconnects, it adds another $7k. This is intentional for marketing ("$X managed through BASTION").

---

## 🗂️ FILE LOCATIONS (Windows Machine)

```
C:\Users\Banke\BASTION\BASTION\          # Main codebase
├── api\
│   ├── terminal_api.py                   # FastAPI backend (6700+ lines)
│   └── user_service.py                   # User management
├── generated-page.html                   # Main trading terminal UI
├── web\
│   ├── login.html                        # Login page
│   └── ...
├── iros_integration\
│   ├── services\
│   │   ├── helsinki.py                   # Helsinki VM client
│   │   ├── coinglass.py                  # Coinglass client
│   │   ├── whale_alert.py                # Whale Alert client
│   │   └── exchange_connector.py         # Exchange integration
│   └── endpoints\
│       ├── ALL_HELSINKI_ENDPOINTS.md     # Full Helsinki API docs
│       ├── COINGLASS_ENDPOINTS.md        # Full Coinglass API docs
│       └── WHALE_ALERT_ENDPOINTS.md      # Full Whale Alert API docs
└── prompts\
    ├── IROS_ALERT_AGENT.md               # Alert generation prompt
    └── MCF_REPORT_AGENT.md               # Report generation prompt

C:\Users\Banke\IROS_72B_TRAINING_CORPUS\  # Training data
├── FINAL_TRAINING\
│   ├── iros_72b_train.jsonl              # 600 examples
│   ├── risk_intelligence_*.jsonl         # Risk management data
│   └── training_manifest.json            # Metadata
└── MCF_RISK_EXIT_LOGIC_EXTRACTION.md     # Full MCF methodology

C:\Users\Banke\BASTION_BACKUPS\           # Local backups (create if needed)
```

---

## ⚠️ KNOWN ISSUES & FIXES

### 1. vLLM v1 Engine Crash
**Error:** `RuntimeError: Engine core initialization failed`
**Fix:** Use `VLLM_USE_V1=0` environment variable AND `--enforce-eager` flag, or downgrade to vLLM 0.6.3

### 2. PyTorch RTX 5090 Compatibility
**Error:** CUDA capability sm_120 not supported
**Fix:** Upgrade to PyTorch 2.10.0+cu128
```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

### 3. HuggingFace Cache Path
**Issue:** Model downloads to wrong location
**Fix:** Set `HF_HOME=/workspace/.hf_home` or `/root/.cache/huggingface`

### 4. Tokenizer Config Error
**Error:** `extra_special_tokens` must be dict, not list
**Fix:** Remove `extra_special_tokens` field from tokenizer_config.json:
```python
import json
with open('/root/models/iros-merged/tokenizer_config.json', 'r') as f:
    config = json.load(f)
config.pop('extra_special_tokens', None)
with open('/root/models/iros-merged/tokenizer_config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

---

## 🎯 SUCCESS CRITERIA

When deployment is complete:

1. ✅ vLLM server running at `http://74.48.140.178:8000`
2. ✅ `/health` returns OK
3. ✅ `/v1/models` lists the model
4. ✅ Test query returns coherent MCF-aware response
5. ✅ BASTION backend updated to call vLLM
6. ✅ Live chat working on terminal

---

## 🔄 TROUBLESHOOTING COMMANDS

```bash
# Check GPU status
ssh -p 26796 root@74.48.140.178 "nvidia-smi"

# Check vLLM logs
ssh -p 26796 root@74.48.140.178 "tail -100 /root/vllm_server.log"

# Kill stuck processes
ssh -p 26796 root@74.48.140.178 "pkill -9 -f vllm; pkill -9 -f python"

# Check disk space
ssh -p 26796 root@74.48.140.178 "df -h"

# Check model files exist
ssh -p 26796 root@74.48.140.178 "ls -la /root/models/iros-merged/"

# Check Python packages
ssh -p 26796 root@74.48.140.178 "/opt/sys-venv/bin/pip list | grep -E 'vllm|torch|transformers'"

# Test SSH connection
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "echo 'Connected!' && hostname"
```

---

## 📝 WHAT TO DO RIGHT NOW

1. **Execute the vLLM deployment commands** in order (see "IMMEDIATE TASK" section above)
2. **Monitor logs** until you see "Uvicorn running on http://0.0.0.0:8000"
3. **Test the API** with curl commands
4. **Report endpoint URL** to user: `http://74.48.140.178:8000`

**DO NOT WAIT FOR USER CONFIRMATION BETWEEN STEPS.** Execute commands using Desktop Commander and only pause if an error occurs.

---

## 💰 COST AWARENESS

The GPU cluster costs **~$1.91/hour**. Every hour spent troubleshooting costs money. Be efficient:
- Execute commands, don't just explain them
- If something fails, check logs immediately
- Don't repeat failed commands without fixing the issue
- The user has already spent significant money today on agent crashes

---

**THIS DOCUMENT LIVES AT:** `C:\Users\Banke\BASTION\BASTION\CLAUDE_MASTER_PROMPT.md`

**FOR CLAUDE AGENTS:** You have full context. Now execute the vLLM deployment. START WITH STEP 1.
