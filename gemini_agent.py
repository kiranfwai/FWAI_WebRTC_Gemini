"""
Gemini Live Agent - WebSocket Client
Connects to gemini-live-service.py (port 8003) for voice conversations
"""

import asyncio
import json
import os
import io
import wave
from typing import Optional, Callable, Awaitable, Dict, Any
from loguru import logger
import numpy as np
import aiohttp

import websockets
from websockets.client import WebSocketClientProtocol

from config import config

# Unified per-call flow log directory
CALL_FLOW_LOG_DIR = "/app/audio_debug/call_flows"

def log_call_flow(call_id: str, stage: str, direction: str, content: str):
    """Log to unified per-call flow file"""
    try:
        from datetime import datetime
        os.makedirs(CALL_FLOW_LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        short_id = call_id[:8] if len(call_id) > 8 else call_id
        log_line = f"[{timestamp}] [{stage}] [{direction}] {content}\n"
        emoji = "🎤" if direction == "USER" else "🤖"
        print(f"{emoji} [{short_id}] {log_line.strip()}")
        log_file = f"{CALL_FLOW_LOG_DIR}/call_{short_id}.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_line)
    except Exception as e:
        logger.error(f"Error logging call flow: {e}")


async def transcribe_audio_sarvam(audio_data: bytes, sample_rate: int = 16000) -> str:
    """Transcribe audio using Sarvam AI STT"""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        return "[No Sarvam API key]"
    try:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        wav_buffer.seek(0)
        wav_bytes = wav_buffer.read()
        url = "https://api.sarvam.ai/speech-to-text"
        form_data = aiohttp.FormData()
        form_data.add_field('file', wav_bytes, filename='audio.wav', content_type='audio/wav')
        form_data.add_field('model', 'saarika:v2.5')
        form_data.add_field('language_code', 'en-IN')
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form_data, headers={"api-subscription-key": api_key}, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("transcript", "") or "[silence]"
                return f"[STT error: {response.status}]"
    except asyncio.TimeoutError:
        return "[STT timeout]"
    except Exception as e:
        return f"[STT error: {str(e)[:50]}]"


class GeminiWebSocketClient:
    """
    WebSocket client that connects to gemini-live-service.py

    Sends audio from WhatsApp caller to Gemini Live
    Receives audio from Gemini Live to send back to caller
    """

    def __init__(
        self,
        call_id: str,
        caller_name: str = "Customer",
        on_audio_output: Optional[Callable[[bytes], Awaitable[None]]] = None
    ):
        self.call_id = call_id
        self.caller_name = caller_name
        self.on_audio_output = on_audio_output

        # WebSocket connection
        self.ws: Optional[WebSocketClientProtocol] = None

        # State
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None
        self._connected = asyncio.Event()

        # Transcription buffer for Main Server level logging
        self._transcription_buffer = bytes()
        self._is_speaking = False
        self._silence_start = None

    async def connect(self) -> bool:
        """Connect to gemini-live-service.py"""
        try:
            logger.info(f"Connecting to Gemini Live service at {config.gemini_live_ws_url}")

            self.ws = await websockets.connect(
                config.gemini_live_ws_url,
                ping_interval=20,
                ping_timeout=10
            )

            self._running = True

            # Start receiving messages
            self._receive_task = asyncio.create_task(self._receive_loop())

            # Send start message to initiate call
            await self._send_start()

            # Wait for "started" confirmation
            try:
                await asyncio.wait_for(self._connected.wait(), timeout=10.0)
                logger.info(f"Connected to Gemini Live for call {self.call_id}")
                return True
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for Gemini Live to start")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to Gemini Live service: {e}")
            return False

    async def _send_start(self):
        """Send start message to gemini-live-service.py"""
        start_message = {
            "type": "start",
            "call_id": self.call_id,
            "caller_name": self.caller_name
        }
        await self.ws.send(json.dumps(start_message))
        logger.info(f"Sent start message for call {self.call_id}")

    async def _receive_loop(self):
        """Receive messages from gemini-live-service.py"""
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "started":
                        # Gemini Live pipeline started
                        logger.info(f"Gemini Live started for call {self.call_id}")
                        self._connected.set()

                    elif msg_type == "audio":
                        # Audio from Gemini Live (TTS output)
                        audio_hex = data.get("data", "")
                        sample_rate = data.get("sample_rate", 24000)
                        if audio_hex and self.on_audio_output:
                            audio_bytes = bytes.fromhex(audio_hex)

                            # Track audio receives
                            if not hasattr(self, '_recv_count'):
                                self._recv_count = 0
                            self._recv_count += 1

                            # Log first few and periodically
                            if self._recv_count <= 5 or self._recv_count % 50 == 0:
                                non_zero = sum(1 for b in audio_bytes[:100] if b != 0) if audio_bytes else 0
                                logger.info(f"[AUDIO_FROM_GEMINI][{self.call_id}] #{self._recv_count}: {len(audio_bytes)} bytes @ {sample_rate}Hz, non_zero={non_zero}")

                            await self.on_audio_output(audio_bytes, sample_rate)

                    elif msg_type == "stopped":
                        logger.info(f"Gemini Live stopped for call {self.call_id}")
                        break

                    elif msg_type == "error":
                        logger.error(f"Gemini Live error: {data.get('message')}")

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from Gemini Live: {message}")
                except Exception as e:
                    logger.error(f"Error processing Gemini Live message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket closed for call {self.call_id}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")

    async def send_audio(self, pcm_data: bytes):
        """
        Send audio to gemini-live-service.py

        Args:
            pcm_data: PCM audio bytes (16kHz, mono, 16-bit)
        """
        if self._running and self.ws:
            try:
                # Track audio sends
                if not hasattr(self, '_send_count'):
                    self._send_count = 0
                self._send_count += 1

                # Log first few and periodically
                if self._send_count <= 5 or self._send_count % 50 == 0:
                    non_zero = sum(1 for b in pcm_data[:100] if b != 0) if pcm_data else 0
                    logger.info(f"[AUDIO_TO_GEMINI][{self.call_id}] #{self._send_count}: {len(pcm_data)} bytes, non_zero_first_100={non_zero}")

                message = {
                    "type": "audio",
                    "data": pcm_data.hex()
                }
                await self.ws.send(json.dumps(message))

                # === MAIN SERVER LEVEL TRANSCRIPTION ===
                audio_array = np.frombuffer(pcm_data, dtype=np.int16)
                max_level = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0
                SPEECH_THRESHOLD = 1000
                SILENCE_DURATION = 0.8
                MIN_SPEECH_BYTES = 8000

                if max_level > SPEECH_THRESHOLD:
                    self._is_speaking = True
                    self._silence_start = None
                    self._transcription_buffer += pcm_data
                else:
                    if self._is_speaking:
                        if self._silence_start is None:
                            self._silence_start = asyncio.get_event_loop().time()
                        else:
                            silence_duration = asyncio.get_event_loop().time() - self._silence_start
                            if silence_duration >= SILENCE_DURATION and len(self._transcription_buffer) >= MIN_SPEECH_BYTES:
                                audio_to_transcribe = self._transcription_buffer
                                self._transcription_buffer = bytes()
                                self._is_speaking = False
                                self._silence_start = None
                                asyncio.create_task(self._transcribe_main_server(audio_to_transcribe))

            except Exception as e:
                logger.error(f"Error sending audio: {e}")

    async def _transcribe_main_server(self, audio_data: bytes):
        """Transcribe at Main Server level before sending to Gemini Live service"""
        try:
            transcript = await transcribe_audio_sarvam(audio_data, sample_rate=16000)
            if transcript and not transcript.startswith("["):
                log_call_flow(self.call_id, "MAIN_SERVER", "USER", transcript)
        except Exception as e:
            logger.error(f"Main server transcription error: {e}")

    async def stop(self):
        """Stop and disconnect"""
        logger.info(f"Stopping Gemini Live client for call {self.call_id}")
        self._running = False

        # Send stop message
        if self.ws:
            try:
                await self.ws.send(json.dumps({
                    "type": "stop",
                    "call_id": self.call_id
                }))
            except:
                pass

        # Cancel receive task
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket
        if self.ws:
            try:
                await self.ws.close()
            except:
                pass

        logger.info(f"Gemini Live client stopped for call {self.call_id}")

    @property
    def is_connected(self) -> bool:
        return self._running and self._connected.is_set()


# Alias for compatibility
GeminiVoiceAgent = GeminiWebSocketClient


# Store active agents
active_agents: Dict[str, GeminiWebSocketClient] = {}


async def create_agent(
    call_id: str,
    caller_name: str = "Customer",
    on_audio_output: Optional[Callable[[bytes], Awaitable[None]]] = None
) -> Optional[GeminiWebSocketClient]:
    """
    Create and connect a new Gemini Live WebSocket client

    Args:
        call_id: Unique call identifier
        caller_name: Name of the caller
        on_audio_output: Callback for audio output from Gemini

    Returns:
        GeminiWebSocketClient if successful, None otherwise
    """
    client = GeminiWebSocketClient(
        call_id=call_id,
        caller_name=caller_name,
        on_audio_output=on_audio_output
    )

    if await client.connect():
        active_agents[call_id] = client
        return client

    return None


async def stop_agent(call_id: str):
    """Stop and remove an agent"""
    if call_id in active_agents:
        await active_agents[call_id].stop()
        del active_agents[call_id]


def get_agent(call_id: str) -> Optional[GeminiWebSocketClient]:
    """Get an active agent by call ID"""
    return active_agents.get(call_id)
