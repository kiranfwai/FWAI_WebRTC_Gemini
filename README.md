# WhatsApp Voice Calling with Gemini Live

AI Voice Agent for WhatsApp Business Voice Calls using Google Gemini Live.

## Architecture

```
+---------------------------------------------------------------------+
|                                                                     |
|  WhatsApp <--WebRTC--> main.py <--WebSocket--> gemini-live-service.py
|   Call                 (port 3000)              (port 8003)         |
|                            |                         |              |
|                       +----+----+              +-----+-----+        |
|                       | aiortc  |              |  Pipecat  |        |
|                       | (audio) |              |  Gemini   |        |
|                       +---------+              |   Live    |        |
|                                                +-----------+        |
+---------------------------------------------------------------------+
```

## Flow

1. **Make Call**: `POST /make-call` -> WhatsApp rings user
2. **User Answers**: WebRTC connection established
3. **Agent Connects**: `main.py` connects to `gemini-live-service.py` via WebSocket
4. **Greeting**: Gemini Live speaks first with greeting prompt
5. **Conversation**: Two-way audio flows:
   - User speaks -> aiortc captures -> WebSocket -> Gemini Live
   - Gemini responds -> WebSocket -> aiortc -> User hears

## Quick Start with Docker (Recommended)

Docker eliminates all dependency installation issues.

### 1. Install Docker

- Windows: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Linux/WSL: `sudo apt install docker.io docker-compose`

### 2. Configure Environment

Edit `.env` with your credentials:
```env
PHONE_NUMBER_ID=your_whatsapp_phone_number_id
META_ACCESS_TOKEN=your_meta_access_token
META_VERIFY_TOKEN=your_verify_token
GOOGLE_API_KEY=your_google_api_key
```

### 3. Start Services

**Windows:**
```cmd
start-docker.bat
```

**Linux/WSL:**
```bash
chmod +x start-docker.sh
./start-docker.sh
```

Or manually:
```bash
docker-compose build
docker-compose up -d
```

### 4. Expose with ngrok

```bash
ngrok http 3000
```

Configure webhooks in Meta Developer Console:
- Messages: `https://your-ngrok-url/webhook`
- Calls: `https://your-ngrok-url/call-events`

### 5. Make a Call

```bash
curl -X POST http://localhost:3000/make-call \
  -H "Content-Type: application/json" \
  -d '{"phoneNumber": "919052034075", "contactName": "Test Customer"}'
```

### View Logs

```bash
docker-compose logs -f
```

### Stop Services

```bash
docker-compose down
```

---

## Manual Installation (Alternative)

If you prefer not to use Docker:

### Prerequisites

- Python 3.11 (not 3.12+ due to dependency issues)
- FFmpeg with development libraries
- Linux/WSL recommended (Windows has compilation issues)

### 1. Install System Dependencies (Ubuntu/WSL)

```bash
sudo apt update
sudo apt install -y \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libopus-dev \
    libvpx-dev \
    gcc \
    g++ \
    pkg-config
```

### 2. Install Python Dependencies

**Terminal 1 - Main Server:**
```bash
cd AI_VOICECALL_Python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Terminal 2 - Gemini Live Service:**
```bash
pip install "pipecat-ai[google]" websockets python-dotenv
```

### 3. Start Services

**Terminal 1 - Gemini Live Service (port 8003):**
```bash
python gemini-live-service.py
```

**Terminal 2 - Main Server (port 3000):**
```bash
python main.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/make-call` | POST | Make outbound call |
| `/webhook` | GET/POST | WhatsApp message webhook |
| `/call-events` | GET/POST | WhatsApp call events webhook |
| `/calls` | GET | List active calls |
| `/calls/{id}/terminate` | POST | End a call |
| `/` | GET | Health check |

## Project Structure

```
AI_VOICECALL_Python/
+-- main.py                  # FastAPI server (port 3000)
+-- gemini-live-service.py   # Gemini Live service (port 8003)
+-- webrtc_handler.py        # WebRTC with aiortc
+-- gemini_agent.py          # WebSocket client to Gemini Live
+-- audio_processor.py       # Audio resampling
+-- whatsapp_client.py       # WhatsApp API client
+-- config.py                # Configuration
+-- requirements.txt         # Python dependencies
+-- Dockerfile               # Main server Docker image
+-- Dockerfile.gemini        # Gemini Live Docker image
+-- docker-compose.yml       # Docker Compose configuration
+-- .env                     # Environment variables
+-- FAWI_Call_BOT.txt        # Conversation script
```

## Why This Architecture?

The Node.js `@roamhq/wrtc` library cannot extract audio from WebRTC tracks. Python's `aiortc` provides full audio access, solving the audio bridge problem.

By keeping `gemini-live-service.py` separate:
- Proven working Gemini Live integration
- Clean separation of concerns
- Easier debugging
- Can restart services independently

## Troubleshooting

### Docker Issues

1. **Port already in use**: Stop existing services or change ports in docker-compose.yml
2. **Build fails**: Ensure Docker has enough memory (at least 4GB)
3. **Connection refused**: Wait for health check to pass before making calls

### Manual Installation Issues

1. **aiortc fails to install**: Use Docker instead, or ensure all FFmpeg dev libraries are installed
2. **Import errors**: Verify Python 3.11 is being used
3. **No audio**: Check logs for WebSocket connection status between main.py and gemini-live-service.py
