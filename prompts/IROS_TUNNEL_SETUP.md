# IROS PERSISTENT TUNNEL SETUP AGENT

## MISSION
Set up a **persistent, always-available tunnel** for the IROS 32B LLM model running on Vast.ai so the BASTION terminal can reliably connect without URL changes.

## CURRENT PROBLEM
- IROS model runs on Vast.ai GPU instance (port 18000)
- Currently using temporary Cloudflare tunnel (`cloudflared tunnel --url http://localhost:18000`)
- **Temporary tunnels generate new URLs on every restart** - breaks production
- BASTION terminal gets 401/connection errors when tunnel URL changes

## SOLUTION OPTIONS (Pick One)

### Option 1: Cloudflare Named Tunnel (RECOMMENDED - FREE)
Named tunnels have **permanent URLs** that don't change.

```bash
# On Vast.ai instance:

# 1. Install cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

# 2. Authenticate (one-time)
cloudflared tunnel login
# This opens browser - login to Cloudflare account

# 3. Create named tunnel (one-time)
cloudflared tunnel create iros-model
# Note the tunnel ID and credentials file path

# 4. Create config file
cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: iros-model
credentials-file: /root/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: iros.yourdomain.com
    service: http://localhost:18000
  - service: http_status:404
EOF

# 5. Route DNS (one-time)
cloudflared tunnel route dns iros-model iros.yourdomain.com

# 6. Run tunnel (add to startup script)
cloudflared tunnel run iros-model
```

**Result:** Permanent URL like `https://iros.yourdomain.com`

### Option 2: Cloudflare Quick Tunnel with Reserved Subdomain
If you don't have a domain, use Cloudflare's trycloudflare.com with a connector:

```bash
# Create a service that auto-reconnects
cat > /etc/systemd/system/iros-tunnel.service << 'EOF'
[Unit]
Description=IROS Cloudflare Tunnel
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/cloudflared tunnel --url http://localhost:18000 --metrics localhost:2000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl enable iros-tunnel
systemctl start iros-tunnel

# Get the URL
journalctl -u iros-tunnel | grep "trycloudflare.com"
```

**Note:** URL still changes on restart, but service auto-reconnects.

### Option 3: ngrok with Reserved Domain ($8/mo)
```bash
# Install ngrok
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# Auth
ngrok config add-authtoken <YOUR_AUTH_TOKEN>

# Reserve domain at ngrok.com dashboard, then:
ngrok http 18000 --domain=iros-model.ngrok.io
```

**Result:** Permanent URL like `https://iros-model.ngrok.io`

### Option 4: Tailscale Funnel (FREE)
```bash
# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# Auth
tailscale up

# Enable HTTPS funnel
tailscale funnel 18000

# Get permanent URL
tailscale funnel status
```

**Result:** Permanent URL like `https://vast-instance.tailnet-name.ts.net`

---

## VAST.AI STARTUP SCRIPT

Add to Vast.ai instance **on-start script** to auto-start tunnel:

```bash
#!/bin/bash

# Wait for model to start
sleep 30

# Start cloudflared tunnel (Option 1 - named tunnel)
cloudflared tunnel run iros-model &

# OR for quick tunnel (Option 2)
# cloudflared tunnel --url http://localhost:18000 &

# Log the URL for debugging
echo "Tunnel started at $(date)" >> /var/log/tunnel.log
```

---

## BASTION INTEGRATION

Once you have a permanent URL, update Railway environment:

```
BASTION_MODEL_URL=https://iros.yourdomain.com
BASTION_MODEL_API_KEY=5c37b5e8e6c2480813aa0cfd4de5c903544b7a000bff729e1c99d9b4538eb34d
```

The API expects OpenAI-compatible endpoints:
- `POST /v1/chat/completions` - Chat completions
- `GET /v1/models` - List models

---

## VERIFICATION STEPS

After setup, verify the tunnel works:

```bash
# Test from local machine
curl -X POST https://YOUR_TUNNEL_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "bastion-32b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'

# Should return JSON with model response, not 401
```

---

## TROUBLESHOOTING

### 401 Unauthorized
- Model API requires auth but tunnel doesn't pass it
- Check if vLLM/model server has `--api-key` flag set
- Verify `BASTION_MODEL_API_KEY` matches model's expected key

### Connection Refused
- Model not running on port 18000
- Check: `curl http://localhost:18000/v1/models`

### Tunnel Disconnects
- Vast.ai instance went to sleep/stopped
- Use systemd service for auto-restart
- Consider "interruptible" vs "on-demand" Vast.ai pricing

### DNS Not Resolving (Named Tunnel)
- Wait 5 minutes for DNS propagation
- Verify: `dig iros.yourdomain.com`

---

## FILES TO CREATE

1. `/etc/systemd/system/iros-tunnel.service` - Auto-start tunnel
2. `~/.cloudflared/config.yml` - Tunnel configuration (named tunnel)
3. Vast.ai on-start script - Initialize tunnel on instance boot

---

## EXPECTED OUTCOME

After completing this setup:
1. IROS model accessible at **permanent URL**
2. Tunnel auto-reconnects on disconnect
3. No manual URL updates needed
4. BASTION terminal reliably connects to IROS

Update `BASTION_MODEL_URL` in Railway with the permanent URL and the Ask BASTION + MCF Reports will work.



