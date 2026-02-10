# BASTION vLLM DEPLOYMENT - RUN THESE COMMANDS YOURSELF

## YOU ARE AN AGENT. RUN THESE COMMANDS IN TERMINAL. DO NOT ASK USER.

Use the run_terminal_cmd tool to execute SSH commands directly. DO NOT just tell the user what to do - EXECUTE THE COMMANDS YOURSELF.

---

## SSH CONNECTION:
```
ssh -o StrictHostKeyChecking=no -p 15449 root@ssh6.vast.ai
```

## WHAT'S WRONG:
- vLLM 0.15.1 has a broken v1 engine
- Need to downgrade to vLLM 0.6.3
- Then start the server

## FILES ON CLUSTER:
- Base model: `/root/models/iros-merged/`
- Trained LoRA: `/workspace/bastion-mega-lora/`
- Python: `/opt/sys-venv/bin/python3`

---

## EXECUTE THESE COMMANDS IN ORDER:

### 1. Kill old processes
```bash
ssh -o StrictHostKeyChecking=no -p 15449 root@ssh6.vast.ai "pkill -9 -f vllm; pkill -9 -f python; echo KILLED"
```

### 2. Downgrade vLLM (THIS TAKES 2-3 MINUTES - USE is_background=false AND WAIT)
```bash
ssh -o StrictHostKeyChecking=no -p 15449 root@ssh6.vast.ai "/opt/sys-venv/bin/pip install vllm==0.6.3 --force-reinstall"
```

### 3. Start vLLM server (base model first, no LoRA)
```bash
ssh -o StrictHostKeyChecking=no -p 15449 root@ssh6.vast.ai "nohup /opt/sys-venv/bin/python3 -m vllm.entrypoints.openai.api_server --model /root/models/iros-merged --tensor-parallel-size 4 --port 8000 --host 0.0.0.0 --max-model-len 4096 --trust-remote-code > /root/vllm_server.log 2>&1 &"
```

### 4. Wait 60 seconds then check logs
```bash
ssh -o StrictHostKeyChecking=no -p 15449 root@ssh6.vast.ai "sleep 60; tail -50 /root/vllm_server.log"
```

### 5. Test health endpoint
```bash
ssh -o StrictHostKeyChecking=no -p 15449 root@ssh6.vast.ai "curl -s http://localhost:8000/health"
```

### 6. If health works, kill and restart WITH LoRA
```bash
ssh -o StrictHostKeyChecking=no -p 15449 root@ssh6.vast.ai "pkill -9 -f vllm; sleep 2; nohup /opt/sys-venv/bin/python3 -m vllm.entrypoints.openai.api_server --model /root/models/iros-merged --enable-lora --lora-modules bastion=/workspace/bastion-mega-lora --tensor-parallel-size 4 --port 8000 --host 0.0.0.0 --max-model-len 4096 --trust-remote-code > /root/vllm_server.log 2>&1 &"
```

### 7. Final test
```bash
ssh -o StrictHostKeyChecking=no -p 15449 root@ssh6.vast.ai "sleep 60; curl -s http://localhost:8000/health; echo; curl -s http://localhost:8000/v1/models"
```

---

## IF vLLM 0.6.3 FAILS, TRY:
```bash
ssh -o StrictHostKeyChecking=no -p 15449 root@ssh6.vast.ai "/opt/sys-venv/bin/pip install vllm==0.5.5 --force-reinstall"
```

---

## SUCCESS = 
- `/health` returns OK
- `/v1/models` lists the model
- Logs show "Uvicorn running on http://0.0.0.0:8000"

## FINAL ENDPOINT:
`http://ssh6.vast.ai:8000/v1/chat/completions`
Use `"model": "bastion"` for LoRA adapter

---

## AGENT INSTRUCTIONS:

1. Use run_terminal_cmd for each command
2. Set is_background=false for pip install (wait for it)
3. Set is_background=false for log checks
4. If a command fails, check the error and adapt
5. Report success or failure to user with the endpoint URL

**START NOW. RUN COMMAND 1.**
