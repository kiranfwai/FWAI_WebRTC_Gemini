# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python-based Voice AI Agent using **Plivo for phone calls** and **Gemini 2.5 Live for AI + native TTS**. This avoids expensive third-party TTS services (like Amazon Polly) by using Gemini's native voice output.

## Commands

```bash
# Create virtual environment and install
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt

# Run server
python run.py

# Expose via ngrok (required for Plivo callbacks)
ngrok http 3001
```

## Project Structure

```
FWAI_WebRTC_Gemini/
├── src/
│   ├── __init__.py
│   ├── app.py                      # FastAPI server with all endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # Configuration management
│   │   └── audio_processor.py      # Audio format conversion
│   ├── services/
│   │   ├── __init__.py
│   │   ├── plivo_gemini_stream.py  # Plivo + Gemini Live bridge (MAIN)
│   │   ├── gemini_live_tts.py      # Gemini Live TTS utility
│   │   ├── gemini_agent.py         # Legacy Gemini WebSocket client
│   │   └── whatsapp_client.py      # WhatsApp Business API client
│   ├── handlers/
│   │   ├── __init__.py
│   │   └── webrtc_handler.py       # WebRTC handling (legacy)
│   └── adapters/
│       ├── plivo_adapter.py        # Plivo API adapter
│       └── ...                     # Other adapters
├── transcripts/                    # Call transcripts (if enabled)
├── audio_cache/                    # Generated audio files
├── logs/                           # Application logs
├── .env                            # Environment configuration
├── requirements.txt
└── run.py                          # Entry point
```

## Architecture

### High-Level Flow: Plivo + Gemini 2.5 Live

```
┌─────────────┐     mulaw 8kHz      ┌─────────────────┐     WebSocket      ┌─────────────────┐
│   Caller    │ ←──────────────────→│     Plivo       │ ←─────────────────→│     Server      │
│  (Phone)    │                     │   (Stream WS)   │                    │  (FastAPI WS)   │
└─────────────┘                     └─────────────────┘                    └────────┬────────┘
                                                                                    │
                                                                           PCM 16kHz (in)
                                                                           PCM 24kHz (out)
                                                                                    │
                                                                                    ↓
                                                                           ┌─────────────────┐
                                                                           │  Gemini 2.5     │
                                                                           │  Live API       │
                                                                           │  (native TTS)   │
                                                                           └─────────────────┘
```

### Inbound Call Flow (User Calls You)

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                 INBOUND CALL FLOW                                         │
└──────────────────────────────────────────────────────────────────────────────────────────┘

  USER'S PHONE                    PLIVO CLOUD                     YOUR SERVER
      │                              │                                 │
      │  1. User dials your          │                                 │
      │     Plivo number             │                                 │
      │─────────────────────────────>│                                 │
      │                              │                                 │
      │                              │  2. POST /plivo/answer          │
      │                              │────────────────────────────────>│
      │                              │                                 │
      │                              │  3. Return <Stream> XML         │
      │                              │<────────────────────────────────│
      │                              │                                 │
      │  4. "Connected to AI..."     │                                 │
      │<─────────────────────────────│                                 │
      │     (Polly TTS - brief)      │                                 │
      │                              │                                 │
      │                              │  5. WebSocket connects          │
      │                              │     /plivo/stream/{uuid}        │
      │                              │════════════════════════════════>│
      │                              │                                 │
      │                              │                    6. Create Gemini Live session
      │                              │                       PlivoGeminiSession
      │                              │                                 │
      │  7. AI Greeting              │                                 │
      │<═════════════════════════════│<════════════════════════════════│
      │  "Hello! I'm Vishnu..."      │         (Gemini Native TTS)     │
      │                              │                                 │
      │  8. Bidirectional audio conversation                           │
      │<═════════════════════════════>│<══════════════════════════════>│
      │                              │                                 │
```

### Outbound Call Flow (You Call User)

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                OUTBOUND CALL FLOW                                         │
└──────────────────────────────────────────────────────────────────────────────────────────┘

  YOUR CLIENT                    YOUR SERVER                    PLIVO CLOUD                USER
      │                              │                              │                        │
      │ 1. POST /plivo/make-call     │                              │                        │
      │ {phoneNumber: "+91..."}      │                              │                        │
      │─────────────────────────────>│                              │                        │
      │                              │                              │                        │
      │                              │ 2. plivo_adapter.make_call() │                        │
      │                              │     POST to Plivo API        │                        │
      │                              │─────────────────────────────>│                        │
      │                              │                              │                        │
      │                              │ 3. {call_uuid: "abc123"}     │                        │
      │                              │<─────────────────────────────│                        │
      │                              │                              │                        │
      │ 4. {success: true,           │                              │  5. Plivo dials user  │
      │     call_uuid: "abc123"}     │                              │───────────────────────>│
      │<─────────────────────────────│                              │                        │
      │                              │                              │  📱 Phone rings...    │
      │                              │                              │                        │
      │                              │                              │  6. User answers      │
      │                              │                              │<───────────────────────│
      │                              │                              │                        │
      │                              │ 7. POST /plivo/answer        │                        │
      │                              │<─────────────────────────────│                        │
      │                              │                              │                        │
      │                              │ 8. Return <Stream> XML       │                        │
      │                              │─────────────────────────────>│                        │
      │                              │                              │                        │
      │                              │                              │  9. "Connected..."    │
      │                              │                              │───────────────────────>│
      │                              │                              │     (Polly - brief)   │
      │                              │                              │                        │
      │                              │ 10. WebSocket connects       │                        │
      │                              │<═════════════════════════════│                        │
      │                              │                              │                        │
      │                              │ 11. Gemini Live session      │                        │
      │                              │     AI greeting audio        │                        │
      │                              │═════════════════════════════>│═══════════════════════>│
      │                              │                              │  🔊 "Hello! I'm..."   │
      │                              │                              │                        │
      │                              │ 12. Bidirectional conversation                        │
      │                              │<════════════════════════════>│<══════════════════════>│
      │                              │                              │                        │
```

### Key Point: Both Flows Converge

```
INBOUND:   User calls → Plivo → /plivo/answer ─────┐
                                                   │
                                                   ▼
                                        ┌─────────────────────┐
                                        │   /plivo/answer     │
                                        │   Returns <Stream>  │
                                        └──────────┬──────────┘
                                                   │
OUTBOUND:  /plivo/make-call → Plivo API →          │
           User answers → Plivo → /plivo/answer ───┘
                                                   │
                                                   ▼
                                        ┌─────────────────────┐
                                        │ /plivo/stream/{uuid}│
                                        │     WebSocket       │
                                        └──────────┬──────────┘
                                                   │
                                                   ▼
                                        ┌─────────────────────┐
                                        │ PlivoGeminiSession  │
                                        │ Gemini 2.5 Live     │
                                        │ Native Audio (Kore) │
                                        └─────────────────────┘
```

## Audio Format Conversions

| Direction | Source Format | Target Format | Conversion |
|-----------|--------------|---------------|------------|
| Caller → Gemini | mulaw 8kHz | PCM 16kHz 16-bit | `audioop.ulaw2lin()` + `ratecv()` |
| Gemini → Caller | PCM 24kHz 16-bit | mulaw 8kHz | `ratecv()` + `audioop.lin2ulaw()` |

## Key Modules

| Module | Purpose |
|--------|---------|
| `src/app.py` | FastAPI server with Plivo endpoints |
| `src/services/plivo_gemini_stream.py` | **Main bridge** - Plivo audio ↔ Gemini Live |
| `src/adapters/plivo_adapter.py` | Plivo API client for outbound calls |
| `src/core/config.py` | Configuration and environment variables |

## API Endpoints

### Plivo + Gemini Live (NEW)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/plivo/make-call` | POST | **Outbound call** - Initiates call via Plivo API |
| `/plivo/answer` | POST | Returns `<Stream>` XML (both inbound & outbound) |
| `/plivo/stream/{call_uuid}` | WebSocket | Bidirectional audio stream |
| `/plivo/stream-status` | POST | Status callbacks from Plivo |
| `/plivo/hangup` | POST | Call hangup callback |

### Legacy (WhatsApp WebRTC)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/make-call` | POST | WhatsApp WebRTC outbound (legacy) |
| `/webhook` | GET/POST | WhatsApp message webhook |
| `/call-events` | GET/POST | WhatsApp call events |

## Making an Outbound Call

```bash
# Using the new Plivo + Gemini Live endpoint
curl -X POST https://your-ngrok-url/plivo/make-call \
  -H "Content-Type: application/json" \
  -d '{
    "phoneNumber": "+919876543210",
    "contactName": "John Doe"
  }'

# Response
{
  "success": true,
  "call_uuid": "abc123-def456",
  "message": "Call initiated to +919876543210. Waiting for user to answer."
}
```

## Key Classes

- **`PlivoGeminiSession`** - Manages a single call session bridging Plivo and Gemini Live
  - Uses `async with client.aio.live.connect()` for proper session management
  - Handles audio conversion between mulaw (Plivo) and PCM (Gemini)
  - Queues audio for smooth streaming

- **`PlivoAdapter`** - Plivo API client
  - `make_call()` - Initiates outbound calls
  - `terminate_call()` - Ends active calls
  - `get_call_details()` - Retrieves call information

## Session Lifecycle

1. **Initiation**:
   - Inbound: User calls your Plivo number
   - Outbound: `POST /plivo/make-call` triggers Plivo API
2. **Answer**: Plivo hits `/plivo/answer` when call connects
3. **Stream**: Server returns `<Stream>` XML → Plivo connects WebSocket
4. **Gemini**: `PlivoGeminiSession` creates Gemini Live 2.5 connection
5. **Greeting**: Gemini sends AI greeting (native TTS - Kore voice)
6. **Conversation**: Bidirectional audio: Plivo ↔ Server ↔ Gemini
7. **Hangup**: User ends call → "stop" event → session cleanup

## Environment Variables

```env
# Plivo
PLIVO_AUTH_ID=<Plivo Console>
PLIVO_AUTH_TOKEN=<Plivo Console>
PLIVO_FROM_NUMBER=<Your Plivo Number>
PLIVO_CALLBACK_URL=<ngrok URL>

# Google
GOOGLE_API_KEY=<Google Cloud / AI Studio>

# Optional
TTS_VOICE=Kore
ENABLE_TRANSCRIPTS=true
DEBUG=true
```

## Important Notes

1. **Gemini Live requires `async with`** - The `client.aio.live.connect()` returns an async context manager, not an awaitable. Always use `async with`.

2. **Audio sample rates matter**:
   - Plivo streams: mulaw 8kHz
   - Gemini input: PCM 16kHz 16-bit mono
   - Gemini output: PCM 24kHz 16-bit mono

3. **ngrok required** - Plivo needs a public URL for WebSocket connections. Run `ngrok http 3001` and update `PLIVO_CALLBACK_URL`.

4. **Plivo Stream XML format**:
```xml
<Response>
    <Speak voice="Polly.Aditi">Connected to AI Assistant. Please wait.</Speak>
    <Stream streamTimeout="86400" keepCallAlive="true" bidirectional="true"
            contentType="audio/x-mulaw;rate=8000"
            statusCallbackUrl="{status_url}">{stream_url}</Stream>
</Response>
```

5. **Both inbound and outbound use same Gemini flow** - After `/plivo/answer`, the flow is identical for both call directions.

## Session Continuation Notes

When continuing work on this project in a new Claude Code session:

### 1. Start the Server


### 2. Verify ngrok is Running


### 3. Update PLIVO_CALLBACK_URL
If ngrok URL changed, update :


### 4. Test a Call
{"detail":[{"type":"json_invalid","loc":["body",1],"msg":"JSON decode error","input":{},"ctx":{"error":"Expecting property name enclosed in double quotes"}}]}

### 5. Check Logs


### Current Working Configuration

- **Model:**  (supports bidiGenerateContent)
- **Audio Format:** L16 16kHz (input) / L16 24kHz (output)
- **Voice:** Kore (Indian English)
- **Prompt:** FWAI Sales Agent (Vishnu) from 

### Known Issues & Solutions

1. **"Model not found for bidiGenerateContent"**
   - Use  model
   - NOT  or 

2. **No audio after "Connected to AI"**
   - Check Google Live API connection in logs
   - Verify  message received
   - Ensure 

3. **Call not connecting**
   - Check ngrok is running
   - Verify PLIVO_CALLBACK_URL matches ngrok URL
   - Check Plivo account has credits

4. **High latency**
   - Reduce BUFFER_SIZE in plivo_gemini_stream.py
   - Current optimized value: 4800
