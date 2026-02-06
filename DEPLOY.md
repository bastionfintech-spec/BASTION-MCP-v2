# BASTION Terminal - Vercel Deployment Guide

## Quick Deploy

1. **Push to GitHub** (already done)
2. **Connect to Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Click "Add New Project"
   - Import the `bastion-terminal` repository
   - Framework Preset: **Other**

3. **Set Environment Variables in Vercel:**
   Go to Project Settings → Environment Variables and add:
   
   | Name | Value |
   |------|-------|
   | `HELSINKI_API_KEY` | Your Helsinki VM API key |
   | `COINGLASS_API_KEY` | Your Coinglass API key |
   | `WHALE_ALERT_API_KEY` | Your Whale Alert API key |

4. **Deploy!**

## Project Structure for Vercel

```
bastion-terminal/
├── api/
│   └── terminal_api.py      # FastAPI serverless function
├── web/
│   ├── visualizations.html  # 3D visualizations page
│   ├── research.html        # Research terminal page
│   └── account.html         # Account center page
├── generated-page.html      # Main trading terminal
├── vercel.json              # Vercel configuration
└── requirements.txt         # Python dependencies
```

## Routes

| Route | Page |
|-------|------|
| `/` | Main Trading Terminal |
| `/visualizations` | 3D Intelligence Lab |
| `/research` | Research Terminal |
| `/account` | Account Center (API Keys) |
| `/api/*` | Backend API endpoints |

## Important Notes

### WebSocket Limitation
Vercel serverless functions **do not support persistent WebSocket connections**. 
The terminal uses polling instead of WebSockets when deployed to Vercel.

Real-time updates work via:
- 1-second price polling
- 5-second data refresh intervals
- API-based updates instead of WebSocket push

### For Full WebSocket Support
If you need real-time WebSocket connections, deploy the backend to:
- **Railway** (recommended, easy Python deployment)
- **Render** (free tier available)
- **Fly.io** (global edge deployment)
- **Your own VPS** (full control)

Then point the frontend to your backend URL.

## Environment Variables

Set these in Vercel dashboard or `.env` locally:

```bash
# Required
HELSINKI_API_KEY=xxx          # CVD, volatility, liquidations
COINGLASS_API_KEY=xxx         # OI, funding, derivatives data
WHALE_ALERT_API_KEY=xxx       # Large transaction monitoring

# Optional
OPENAI_API_KEY=xxx            # For enhanced AI responses
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python api/terminal_api.py

# Access at http://localhost:8888
```

## Custom Domain

1. Go to Vercel Project → Settings → Domains
2. Add your custom domain
3. Update DNS records as instructed

## Troubleshooting

### API calls failing
- Check environment variables are set in Vercel
- Check Vercel function logs for errors

### Slow initial load
- First request after inactivity triggers cold start
- Subsequent requests are fast

### CORS issues
- The API has permissive CORS settings
- If issues persist, check browser console for specific errors



