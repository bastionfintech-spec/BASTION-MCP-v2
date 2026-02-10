# BASTION PRODUCTION DEPLOYMENT - AGENT INSTRUCTIONS

## ⚠️ PRIORITY 1: BACKUP LORA ADAPTERS TO PC FIRST!

Before doing ANYTHING else, backup the trained model to the user's Windows PC.

### Cluster Info:
- **SSH:** `ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178`
- **Python:** `/opt/sys-venv/bin/python3`
- **4x RTX 5090 GPUs**

### What exists on cluster:
- Merged model: `/root/models/iros-merged` (~65GB)
- Trained LoRA: `/root/checkpoints/risk-intelligence-lora`

### BACKUP COMMANDS (run from Windows PowerShell):

```powershell
# Create backup folder
mkdir C:\Users\Banke\BASTION_BACKUPS -Force

# Download the trained LoRA adapter (THIS IS THE IMPORTANT ONE)
scp -o StrictHostKeyChecking=no -P 26796 -r root@74.48.140.178:/root/checkpoints/risk-intelligence-lora C:\Users\Banke\BASTION_BACKUPS\

# Download merged model config files (small, for reference)
scp -o StrictHostKeyChecking=no -P 26796 root@74.48.140.178:/root/models/iros-merged/config.json C:\Users\Banke\BASTION_BACKUPS\
scp -o StrictHostKeyChecking=no -P 26796 "root@74.48.140.178:/root/models/iros-merged/tokenizer*" C:\Users\Banke\BASTION_BACKUPS\

# Verify backup exists
dir C:\Users\Banke\BASTION_BACKUPS
```

**⛔ DO NOT PROCEED UNTIL BACKUP IS CONFIRMED ON LOCAL PC**

---

## PRIORITY 2: INSTALL vLLM

vLLM is a production inference server. It keeps the model loaded in GPU memory and handles multiple user requests efficiently.

```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "/opt/sys-venv/bin/pip install vllm"
```

---

## PRIORITY 3: START vLLM SERVER

This starts the model server using all 4 GPUs in tensor-parallel mode:

```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "nohup /opt/sys-venv/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /root/models/iros-merged \
    --tensor-parallel-size 4 \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 4096 \
    --trust-remote-code \
    > /root/vllm_server.log 2>&1 &"
```

Monitor startup (wait for "Uvicorn running on http://0.0.0.0:8000"):
```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "tail -f /root/vllm_server.log"
```

---

## PRIORITY 4: TEST THE API

Once vLLM shows "running", test it:

```bash
ssh -o StrictHostKeyChecking=no -p 26796 root@74.48.140.178 "curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
        \"model\": \"/root/models/iros-merged\",
        \"messages\": [{\"role\": \"user\", \"content\": \"What is MCF scoring methodology?\"}],
        \"max_tokens\": 200
    }' | head -c 500"
```

---

## HOW THE 4 GPUs WORK TOGETHER:

```
┌─────────────────────────────────────────────────────────┐
│                    vLLM SERVER                          │
│              (Tensor Parallel Mode)                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   The 32B model is SPLIT across all 4 GPUs:             │
│                                                         │
│   GPU 0: Layers 0-15   (~16GB)                          │
│   GPU 1: Layers 16-31  (~16GB)                          │
│   GPU 2: Layers 32-47  (~16GB)                          │
│   GPU 3: Layers 48-63  (~16GB)                          │
│                                                         │
│   ALL 4 GPUs work on EVERY request together             │
│   = Faster inference than single GPU                    │
│   = Can handle multiple concurrent users                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## API ENDPOINTS (OpenAI-compatible):

Once running, vLLM exposes these at `http://74.48.140.178:8000`:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | Chat with BASTION |
| `/v1/completions` | POST | Raw text completion |
| `/v1/models` | GET | List loaded models |
| `/health` | GET | Health check |

---

## CONNECTING BASTION TERMINAL:

The BASTION FastAPI backend should call vLLM like this:

```python
import httpx

VLLM_URL = "http://74.48.140.178:8000/v1/chat/completions"

async def ask_bastion(user_message: str, system_prompt: str = None):
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
```

---

## SYSTEM PROMPTS TO USE:

### For Chat/Analysis:
```
You are BASTION - an institutional-grade crypto trading AI with access to Helsinki VM (33 endpoints), Coinglass Premium (liquidations, funding, OI), and Whale Alert (on-chain tracking). Provide comprehensive market analysis combining multiple data sources.
```

### For Risk Intelligence (position management):
```
You are BASTION Risk Intelligence - an autonomous trade management AI. You monitor live positions and make execution decisions. Output JSON with action, reasoning, and confidence. PRIORITY ORDER: 1) Hard Stop 2) Safety Net 3) Guarding Line 4) Targets 5) Trail 6) Time Exit.
```

---

## ORDER OF OPERATIONS:

1. ⬜ BACKUP LoRA to Windows PC (CRITICAL - DO FIRST!)
2. ⬜ Verify backup exists locally
3. ⬜ Install vLLM on cluster
4. ⬜ Start vLLM server (tensor-parallel on 4 GPUs)
5. ⬜ Wait for server to load model (~2-5 min)
6. ⬜ Test API with curl
7. ⬜ Confirm working, provide endpoint URL to user

---

## TROUBLESHOOTING:

### If vLLM fails to start:
```bash
# Check logs
ssh -p 26796 root@74.48.140.178 "cat /root/vllm_server.log"

# Check GPU memory
ssh -p 26796 root@74.48.140.178 "nvidia-smi"

# Kill stuck processes
ssh -p 26796 root@74.48.140.178 "pkill -f vllm"
```

### If out of memory:
Add `--gpu-memory-utilization 0.85` to vLLM command

---

## SUCCESS CRITERIA:

✅ LoRA backed up to `C:\Users\Banke\BASTION_BACKUPS\`
✅ vLLM server running on port 8000
✅ API responds to test query
✅ User has endpoint URL: `http://74.48.140.178:8000`

---

**START WITH BACKUP! The trained LoRA is irreplaceable!**

