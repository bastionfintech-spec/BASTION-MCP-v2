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
    
    return {"results": results}

