"""
Simple test endpoint for Vercel debugging
"""
from fastapi import FastAPI

app = FastAPI()

@app.get("/api/test")
def test():
    return {"status": "ok", "message": "Vercel Python is working!"}

@app.get("/api/test/imports")
def test_imports():
    results = {}
    
    try:
        import httpx
        results["httpx"] = "ok"
    except Exception as e:
        results["httpx"] = str(e)
    
    try:
        from dotenv import load_dotenv
        results["dotenv"] = "ok"
    except Exception as e:
        results["dotenv"] = str(e)
    
    try:
        import sys
        from pathlib import Path
        bastion_path = Path(__file__).parent.parent
        sys.path.insert(0, str(bastion_path))
        from iros_integration.config.settings import settings
        results["settings"] = f"ok - coinglass: {settings.coinglass.api_key[:8]}..."
    except Exception as e:
        results["settings"] = str(e)
    
    # Test the clients
    try:
        from iros_integration.services.helsinki import HelsinkiClient
        results["helsinki_client"] = "ok"
    except Exception as e:
        results["helsinki_client"] = str(e)
    
    try:
        from iros_integration.services.coinglass import CoinglassClient
        results["coinglass_client"] = "ok"
    except Exception as e:
        results["coinglass_client"] = str(e)
    
    try:
        from iros_integration.services.whale_alert import WhaleAlertClient
        results["whale_alert_client"] = "ok"
    except Exception as e:
        results["whale_alert_client"] = str(e)
    
    try:
        from iros_integration.services.query_processor import QueryProcessor
        results["query_processor"] = "ok"
    except Exception as e:
        results["query_processor"] = str(e)
    
    try:
        from iros_integration.services.exchange_connector import user_context
        results["exchange_connector"] = "ok"
    except Exception as e:
        results["exchange_connector"] = str(e)
    
    # Try importing the main terminal_api
    try:
        from api.terminal_api import app as terminal_app
        results["terminal_api"] = "ok"
    except Exception as e:
        results["terminal_api"] = str(e)
    
    return {"results": results}

