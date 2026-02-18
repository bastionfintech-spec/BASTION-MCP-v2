#!/bin/bash
# BASTION v3 Full Deploy Pipeline
# Run this on the GPU cluster after SSH is restored
# Usage: bash /workspace/deploy_v3.sh

set -e

echo "=========================================="
echo "BASTION v3 Full Deploy Pipeline"
echo "=========================================="

# Step 1: Stop vLLM
echo "[1/5] Stopping vLLM..."
pkill -f vllm || true
sleep 10
echo "  vLLM stopped."

# Step 2: Run fine-tune
echo "[2/5] Running fine-tune..."
/opt/sys-venv/bin/python3 /workspace/finetune_v3.py
echo "  Fine-tune complete."

# Step 3: Start vLLM with new model
echo "[3/5] Starting vLLM with v3 model..."
tmux kill-session -t vllm 2>/dev/null || true
tmux new-session -d -s vllm '/opt/sys-venv/bin/python3 -m vllm.entrypoints.openai.api_server --model /root/models/bastion-risk-v3-merged --served-model-name bastion-32b --tensor-parallel-size 4 --max-model-len 8192 --host 0.0.0.0 --port 8000'
echo "  vLLM starting in tmux session 'vllm'..."

# Step 4: Wait for vLLM to be ready
echo "[4/5] Waiting for vLLM to be ready..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/v1/models | grep -q bastion-32b; then
        echo "  vLLM ready after ${i}0 seconds!"
        break
    fi
    sleep 10
done

# Step 5: Set up port forwarding
echo "[5/5] Setting up port forwarding..."
tmux kill-session -t socat 2>/dev/null || true
tmux new-session -d -s socat 'socat TCP-LISTEN:6006,fork,reuseaddr TCP:localhost:8000'
echo "  Port forwarding active (6006 -> 8000)."

echo ""
echo "=========================================="
echo "DEPLOY COMPLETE"
echo "=========================================="
echo "Model: bastion-risk-v3-merged"
echo "API: http://localhost:8000/v1/chat/completions"
echo "External: port 6006 -> external mapped port"
echo ""
echo "Test with:"
echo '  curl http://localhost:8000/v1/models'
