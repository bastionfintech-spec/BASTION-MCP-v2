"""
BASTION API Entry Point
=======================

Starts the FastAPI server for BASTION risk management API.

Usage:
    python run.py

Server will start on http://localhost:8001
"""

import uvicorn
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8003"))
    
    print(f"""
    +-----------------------------------------------+
    |                                               |
    |              BASTION API Server               |
    |                                               |
    |      Proactive Risk Management                |
    |                                               |
    +-----------------------------------------------+
    
    Starting server on http://{host}:{port}
    
    Endpoints:
      - GET  /health        Health check
      - POST /calculate     Calculate risk levels
      - GET  /docs          API documentation
      
    Press CTRL+C to stop
    """)
    
    uvicorn.run(
        "api.terminal_api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

