# WhatsApp Voice Calling with Gemini Live

AI Voice Agent for WhatsApp Business Voice Calls using Google Gemini Live.

**Now with multi-provider support: WhatsApp, Plivo, and Exotel**

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌─────────────┐                                                            │
│  │  WhatsApp   │───┐                                                        │
│  └─────────────┘   │                                                        │
│                    │     ┌──────────────┐      ┌─────────────────────┐      │
│  ┌─────────────┐   │     │              │      │                     │      │
│  │   Plivo     │───┼────►│   src/app.py    │◄────►│ gemini-live-service │      │
│  └─────────────┘   │     │  (port 3000) │      │    (port 8003)      │      │
│                    │     │              │      │                     │      │
│  ┌─────────────┐   │     └──────┬───────┘      └──────────┬──────────┘      │
│  │   Exotel    │───┘            │                         │                 │
│  └─────────────┘                │                         │                 │
│                           ┌─────┴─────┐             ┌─────┴─────┐           │
│                           │  adapters │             │  Pipecat  │           │
│                           │  + aiortc │             │  Gemini   │           │
│                           └───────────┘             │   Live    │           │
│                                                     └───────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Provider Comparison

| Feature | WhatsApp | Plivo | Exotel |
|---------|----------|-------|--------|
| Outbound Calls | ✅ | ✅ | ✅ |
| Incoming Calls | ✅ | ✅ | ✅ |
| WebRTC | ✅ | ❌ (PSTN) | ❌ (PSTN) |
| India Numbers | Via Business | Requires Compliance | Easy |
| TTS on Call | ❌ | ✅ | Via ExoML |
| Recording | ❌ | ✅ | ✅ (auto) |
| SMS | ✅ | ❌ | ✅ |

## Flow

1. **Make Call**: `POST /make-call` → Provider rings user
2. **User Answers**: Connection established (WebRTC or PSTN)
3. **Agent Connects**: `src/app.py` connects to `gemini-live-service.py` via WebSocket
4. **Greeting**: Gemini Live speaks first with greeting prompt
5. **Conversation**: Two-way audio flows:
   - User speaks → Audio captured → WebSocket → Gemini Live
   - Gemini responds → WebSocket → Audio output → User hears

## Quick Start

### 1. Install Dependencies

```bash
cd /home/kiran/FWAI_WebRTC_Gemini/FWAI_WebRTC_Gemini
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env`:
```env
# Choose provider: whatsapp, plivo, or exotel
CALL_PROVIDER=whatsapp

# WhatsApp Config
PHONE_NUMBER_ID=your_whatsapp_phone_number_id
META_ACCESS_TOKEN=your_meta_access_token
META_VERIFY_TOKEN=your_verify_token

# Plivo Config (if using Plivo)
PLIVO_AUTH_ID=your_plivo_auth_id
PLIVO_AUTH_TOKEN=your_plivo_auth_token
PLIVO_PHONE_NUMBER=+1234567890
PLIVO_CALLBACK_URL=https://your-server.com

# Exotel Config (if using Exotel)
EXOTEL_API_KEY=your_exotel_api_key
EXOTEL_API_TOKEN=your_exotel_api_token
EXOTEL_SID=your_exotel_sid
EXOTEL_CALLER_ID=your_exotel_virtual_number

# Google Gemini
GOOGLE_API_KEY=your_google_api_key
GEMINI_LIVE_PORT=8003
```

### 3. Start Services

**Terminal 1 - Gemini Live Service (port 8003):**
```bash
python src/services/gemini-live-service.py
```

**Terminal 2 - Main Server (port 3000):**
```bash
python src/app.py
```

### 4. Expose with ngrok

```bash
ngrok http 3000
```

Configure webhooks in respective provider console:
- **WhatsApp**: `https://your-ngrok-url/webhook` and `/call-events`
- **Plivo**: `https://your-ngrok-url/plivo/answer` and `/plivo/hangup`
- **Exotel**: `https://your-ngrok-url/exotel/status`

### 5. Make a Call

```bash
curl -X POST http://localhost:3000/make-call \
  -H "Content-Type: application/json" \
  -d '{"phoneNumber": "919052034075", "contactName": "Test Customer"}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/make-call` | POST | Make outbound call |
| `/webhook` | GET/POST | WhatsApp message webhook |
| `/call-events` | GET/POST | WhatsApp call events webhook |
| `/plivo/answer` | POST | Plivo answer webhook |
| `/plivo/hangup` | POST | Plivo hangup webhook |
| `/exotel/status` | POST | Exotel status callback |
| `/calls` | GET | List active calls |
| `/calls/{id}/terminate` | POST | End a call |
| `/` | GET | Health check |

## Project Structure

```
FWAI_WebRTC_Gemini/
├── run.py                  # Entry point
├── requirements.txt
├── .env / .env.example
├── src/
│   ├── app.py              # FastAPI server
│   ├── core/               # config.py, audio_processor.py
│   ├── services/           # gemini_agent.py, whatsapp_client.py
│   ├── handlers/           # webrtc_handler.py
│   └── adapters/           # whatsapp, plivo, exotel adapters
├── logs/                  # Application logs
├── scripts/              # start.sh, start.bat
└── docs/                 # Documentation
```

## Adapter Usage

```python
from src.adapters import WhatsAppAdapter, PlivoAdapter, ExotelAdapter

# WhatsApp
wa = WhatsAppAdapter()
await wa.make_call(phone_number="919052034075", sdp_offer="...")

# Plivo
plivo = PlivoAdapter()
await plivo.make_call(phone_number="+919052034075")
await plivo.speak_text(call_id, "Hello from AI")

# Exotel
exotel = ExotelAdapter()
await exotel.make_call(phone_number="919052034075")
await exotel.connect_two_numbers(from_number="agent", to_number="customer")
```

## Why This Architecture?

The Node.js `@roamhq/wrtc` library couldn't extract audio from WebRTC tracks. Python's `aiortc` provides full audio access, solving the audio bridge problem.

**Multi-provider adapters** allow:
- Easy switching between providers via `CALL_PROVIDER` env var
- Consistent interface across all providers
- Provider-specific features (TTS, recording, etc.)
- Fallback options for different regions

## Plivo + Google Live API (Recommended)

The simplest and most cost-effective setup using Plivo for telephony and Google Live API for real-time AI voice.

### Cost Breakdown
| Component | Provider | Cost |
|-----------|----------|------|
| Phone Line | Plivo (Outbound) | ~/bin/bash.012/min |
| Real-time Audio | Plivo Stream | ~/bin/bash.003/min |
| AI Brain + Voice | Gemini 2.0 Flash | Free tier available |
| Orchestration | Your server | Minimal |

### Quick Test

1. **Start ngrok:**


2. **Update :**
SHELL=/bin/bash
WSL2_GUI_APPS_ENABLED=1
WSL_DISTRO_NAME=Ubuntu-22.04
NAME=DESKTOP-TRROJ24
PWD=/mnt/c
LOGNAME=kiran
HOME=/home/kiran
LANG=C.UTF-8
WSL_INTEROP=/run/WSL/27060_interop
WAYLAND_DISPLAY=wayland-0
TERM=xterm-256color
USER=kiran
DISPLAY=:0
SHLVL=1
XDG_RUNTIME_DIR=/run/user/1000/
WSLENV=
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/lib/wsl/lib:/mnt/c/Users/Admin/bin:/mnt/d/Softwares/Git/mingw64/bin:/mnt/d/Softwares/Git/usr/local/bin:/mnt/d/Softwares/Git/usr/bin:/mnt/d/Softwares/Git/usr/bin:/mnt/d/Softwares/Git/mingw64/bin:/mnt/d/Softwares/Git/usr/bin:/mnt/c/Users/Admin/bin:/mnt/c/Python314/Scripts:/mnt/c/Python314:/mnt/c/Windows/system32:/mnt/c/Windows:/mnt/c/Windows/System32/Wbem:/mnt/c/Windows/System32/WindowsPowerShell/v1.0:/mnt/c/Windows/System32/OpenSSH:/mnt/d/Softwares/Git/cmd:/mnt/d/Softwares:/mnt/c/ProgramData/chocolatey/bin:/Docker/host/bin:/mnt/c/Python314/Scripts:/mnt/c/Python314:/mnt/c/Windows/system32:/mnt/c/Windows:/mnt/c/Windows/System32/Wbem:/mnt/c/Windows/System32/WindowsPowerShell/v1.0:/mnt/c/Windows/System32/OpenSSH:/mnt/d/Softwares/Git/cmd:/mnt/d/Softwares:/mnt/c/ProgramData/chocolatey/bin:/mnt/c/Users/Admin/AppData/Local/Microsoft/WindowsApps:/mnt/c/Users/Admin/AppData/Local/Programs/cursor/resources/app/bin:/mnt/c/Users/Admin/AppData/Roaming/npm:/mnt/c/Users/Admin/AppData/Roaming/npm:/mnt/c/Users/Admin/AppData/Local/Programs/Microsoft VS Code/bin:/mnt/c/Users/Admin/AppData/Local/Programs/Antigravity/bin:/mnt/d/Softwares/Git/usr/bin/vendor_perl:/mnt/d/Softwares/Git/usr/bin/core_perl
DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
HOSTTYPE=x86_64
PULSE_SERVER=unix:/mnt/wslg/PulseServer
_=/usr/bin/env

3. **Start server:**
2026-01-24 21:26:57 | INFO     | src.app:lifespan:68 - ============================================================
2026-01-24 21:26:57 | INFO     | src.app:lifespan:69 - WhatsApp Voice Calling with Gemini Live
2026-01-24 21:26:57 | INFO     | src.app:lifespan:70 - ============================================================
2026-01-24 21:26:57 | INFO     | src.app:lifespan:78 - Server starting on http://0.0.0.0:3000
2026-01-24 21:26:57 | INFO     | src.app:lifespan:79 - Gemini Voice: Kore
2026-01-24 21:26:57 | INFO     | src.app:lifespan:84 - Server shutting down...

4. **Make a call:**
{"detail":[{"type":"json_invalid","loc":["body",1],"msg":"JSON decode error","input":{},"ctx":{"error":"Expecting property name enclosed in double quotes"}}]}

### Transcripts

Call transcripts with timestamps are saved to:


Format:


### Key Files

| File | Purpose |
|------|---------|
|  | FastAPI server with Plivo endpoints |
|  | Plivo ↔ Google Live API bridge |
|  | FWAI sales agent prompt |
|  | Call transcripts with timestamps |
