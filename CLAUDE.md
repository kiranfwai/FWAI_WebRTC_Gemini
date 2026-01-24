# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python-based WhatsApp Business Voice Calling with Gemini Live AI agent. Uses aiortc for WebRTC with full audio access, solving the audio bridge limitation in the Node.js implementation.

## Commands

```bash
# Create virtual environment and install
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt

# Run server
python run.py

# Or use startup scripts
./scripts/start.sh    # Linux/Mac
scripts\start.bat     # Windows
```

## Project Structure

```
FWAI_WebRTC_Gemini/
├── src/
│   ├── __init__.py
│   ├── app.py              # FastAPI server with endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py       # Configuration management
│   │   └── audio_processor.py  # Audio format conversion
│   ├── services/
│   │   ├── __init__.py
│   │   ├── gemini_agent.py     # Gemini Live WebSocket client
│   │   ├── gemini-live-service.py  # Standalone Gemini service
│   │   └── whatsapp_client.py  # WhatsApp Business API client
│   ├── handlers/
│   │   ├── __init__.py
│   │   └── webrtc_handler.py   # WebRTC handling with aiortc
│   └── adapters/
│       ├── __init__.py
│       ├── base.py             # Abstract adapter interface
│       ├── whatsapp_adapter.py
│       ├── plivo_adapter.py
│       └── exotel_adapter.py
├── logs/                   # Application logs
├── scripts/
│   ├── start.sh
│   └── start.bat
├── docs/
│   └── FAWI_Call_BOT.txt
├── .env                    # Environment configuration
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── CLAUDE.md
└── run.py                  # Entry point
```

## Architecture

```
WhatsApp Call ←→ aiortc WebRTC ←→ AudioProcessor ←→ Gemini Live (WebSocket)
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/app.py` | FastAPI server with `/make-call`, `/webhook`, `/call-events` endpoints |
| `src/handlers/webrtc_handler.py` | WebRTC handling with aiortc, CallSession management |
| `src/services/gemini_agent.py` | Gemini Live voice agent WebSocket client |
| `src/core/audio_processor.py` | Audio conversion between WebRTC (48kHz) and Gemini (16kHz) |
| `src/services/whatsapp_client.py` | WhatsApp Business API client |
| `src/core/config.py` | Configuration and environment variable loading |

### Audio Flow

1. **User → Agent**: `WebRTC AudioFrame` → `AudioProcessor.process_webrtc_frame()` → PCM bytes → `GeminiVoiceAgent.feed_audio()`
2. **Agent → User**: Gemini TTSAudioRawFrame → `AudioOutputTrack.feed_audio()` → `WebRTC AudioFrame`

### Key Classes

- `CallSession` - Manages WebRTC peer connection and audio processing for a call
- `GeminiVoiceAgent` - WebSocket client for Gemini Live with audio input/output
- `AudioOutputTrack` - Custom aiortc track that outputs Gemini audio to WebRTC

## Environment Variables

```env
PHONE_NUMBER_ID=<WhatsApp Business Manager>
META_ACCESS_TOKEN=<Meta Developer Console>
META_VERIFY_TOKEN=<webhook verification>
GOOGLE_API_KEY=<Google Cloud>
TTS_VOICE=Kore
```
