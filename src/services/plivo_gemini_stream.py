# Plivo + Google Live API Stream Handler
import asyncio
import json
import base64
from typing import Dict
from loguru import logger
from datetime import datetime
from pathlib import Path
import websockets
from src.core.config import config

# Load FWAI prompt
def load_fwai_prompt():
    try:
        prompts_file = Path(__file__).parent.parent.parent / "prompts.json"
        with open(prompts_file) as f:
            prompts = json.load(f)
            return prompts.get("FWAI_Core", {}).get("prompt", "You are a helpful AI assistant.")
    except:
        return "You are a helpful AI assistant."

FWAI_PROMPT = load_fwai_prompt()

class PlivoGeminiSession:
    def __init__(self, call_uuid: str, caller_phone: str, plivo_ws):
        self.call_uuid = call_uuid
        self.caller_phone = caller_phone
        self.plivo_ws = plivo_ws
        self.goog_live_ws = None
        self.is_active = False
        self.start_streaming = False
        self.stream_id = ""
        self._session_task = None
        self.BUFFER_SIZE = 800  # Reduced for lower latency (was 3200)
        self.inbuffer = bytearray(b"")
        self.greeting_sent = False

    def _save_transcript(self, role, text):
        if not config.enable_transcripts:
            return
        try:
            transcript_dir = Path(__file__).parent.parent.parent / "transcripts"
            transcript_dir.mkdir(exist_ok=True)
            transcript_file = transcript_dir / f"{self.call_uuid}.txt"
            timestamp = datetime.now().strftime("%H:%M:%S")
            with open(transcript_file, "a") as f:
                f.write(f"[{timestamp}] {role}: {text}\n")
            logger.info(f"TRANSCRIPT [{role}]: {text}")
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")

    async def start(self):
        try:
            logger.info(f"Starting Google Live API session for call {self.call_uuid}")
            self.is_active = True
            self._save_transcript("SYSTEM", "Call started")
            self._session_task = asyncio.create_task(self._run_google_live_session())
            return True
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            return False

    async def _run_google_live_session(self):
        url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={config.google_api_key}"
        try:
            async with websockets.connect(url) as ws:
                self.goog_live_ws = ws
                logger.info("Connected to Google Live API")
                await self._send_session_setup()
                async for message in ws:
                    if not self.is_active:
                        break
                    await self._receive_from_google(message)
        except Exception as e:
            logger.error(f"Google Live error: {e}")
        finally:
            self.goog_live_ws = None
            logger.info("Google Live session ended")

    async def _send_session_setup(self):
        msg = {
            "setup": {
                "model": "models/gemini-2.0-flash-exp",
                "generation_config": {
                    "response_modalities": "audio",
                    "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Charon"}}}
                },
                "system_instruction": {"parts": [{"text": FWAI_PROMPT}]},
                "tools": []
            }
        }
        await self.goog_live_ws.send(json.dumps(msg))
        logger.info("Sent session setup with FWAI prompt")

    async def _send_initial_greeting(self):
        """Send initial trigger to make AI greet immediately"""
        if self.greeting_sent or not self.goog_live_ws:
            return
        self.greeting_sent = True
        msg = {
            "client_content": {
                "turns": [{
                    "role": "user",
                    "parts": [{"text": "Start"}]
                }],
                "turn_complete": True
            }
        }
        await self.goog_live_ws.send(json.dumps(msg))
        logger.info("Sent initial greeting trigger")



    async def _receive_from_google(self, message):
        try:
            resp = json.loads(message)
            if "setupComplete" in resp:
                logger.info("Google Live setup complete - AI Ready")
                self.start_streaming = True
                self._save_transcript("SYSTEM", "AI ready")
                # Trigger immediate greeting
                await self._send_initial_greeting()
            if "serverContent" in resp:
                sc = resp["serverContent"]
                if "interrupted" in sc:
                    await self.plivo_ws.send_text(json.dumps({"event": "clearAudio", "stream_id": self.stream_id}))
                if "modelTurn" in sc:
                    parts = sc.get("modelTurn", {}).get("parts", [])
                    for p in parts:
                        if p.get("inlineData", {}).get("data"):
                            audio = p["inlineData"]["data"]
                            await self.plivo_ws.send_text(json.dumps({
                                "event": "playAudio",
                                "media": {"contentType": "audio/x-l16", "sampleRate": 24000, "payload": audio}
                            }))
                        if p.get("text"):
                            self._save_transcript("VISHNU", p["text"])
        except Exception as e:
            logger.error(f"Error: {e}")

    async def handle_plivo_audio(self, audio_b64):
        try:
            if not self.is_active or not self.start_streaming or not self.goog_live_ws:
                return
            chunk = base64.b64decode(audio_b64)
            self.inbuffer.extend(chunk)
            while len(self.inbuffer) >= self.BUFFER_SIZE:
                ac = self.inbuffer[:self.BUFFER_SIZE]
                msg = {"realtime_input": {"media_chunks": [{"mime_type": "audio/pcm;rate=16000", "data": base64.b64encode(bytes(ac)).decode()}]}}
                await self.goog_live_ws.send(json.dumps(msg))
                self.inbuffer = self.inbuffer[self.BUFFER_SIZE:]
        except Exception as e:
            logger.error(f"Audio error: {e}")

    async def handle_plivo_message(self, message):
        event = message.get("event")
        if event == "media":
            payload = message.get("media", {}).get("payload", "")
            if payload:
                await self.handle_plivo_audio(payload)
        elif event == "start":
            self.stream_id = message.get("start", {}).get("streamId", "")
            logger.info(f"Stream started: {self.stream_id}")
        elif event == "stop":
            await self.stop()

    async def stop(self):
        logger.info(f"Stopping session for {self.call_uuid}")
        self.is_active = False
        if self.goog_live_ws:
            try:
                await self.goog_live_ws.close()
            except:
                pass
        if self._session_task:
            self._session_task.cancel()
        self._save_transcript("SYSTEM", "Call ended")

_sessions: Dict[str, PlivoGeminiSession] = {}

async def create_session(call_uuid, caller_phone, plivo_ws):
    session = PlivoGeminiSession(call_uuid, caller_phone, plivo_ws)
    if await session.start():
        _sessions[call_uuid] = session
        return session
    return None

async def get_session(call_uuid):
    return _sessions.get(call_uuid)

async def remove_session(call_uuid):
    if call_uuid in _sessions:
        await _sessions[call_uuid].stop()
        del _sessions[call_uuid]
