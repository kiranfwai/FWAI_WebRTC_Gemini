#!/usr/bin/env python3
"""
Entry point for WhatsApp Voice Calling with Gemini Live

Run with: python run.py
"""

import uvicorn
from src.core.config import config

if __name__ == "__main__":
    uvicorn.run(
        "src.app:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="debug" if config.debug else "info"
    )
