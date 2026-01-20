"""
Gemini Live Service - Native Audio Streaming
Sends audio directly to Gemini Live, receives audio responses
No separate STT/LLM/TTS - Gemini Live handles everything natively
"""

import os
import asyncio
import json
import sys
import base64
import aiohttp
from datetime import datetime
from collections import deque
import io
import wave

# Transcript log file
TRANSCRIPT_LOG_FILE = "/app/audio_debug/call_transcripts.log"

# Unified per-call flow log directory
CALL_FLOW_LOG_DIR = "/app/audio_debug/call_flows"

# Setup logging
def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def log_call_flow(call_id: str, stage: str, direction: str, content: str):
    """Log to unified per-call flow file"""
    try:
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
        log(f"Error logging call flow: {e}")


def log_transcript(call_id: str, speaker: str, text: str):
    """Log transcript to file for both user and agent speech"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        short_id = call_id[:8] if len(call_id) > 8 else call_id
        log_line = f"[{timestamp}] [{short_id}] {speaker}: {text}\n"

        # Also print to console with highlighting
        if speaker == "AGENT":
            print(f"🤖 {log_line.strip()}")
        else:
            print(f"👤 {log_line.strip()}")

        # Write to file
        with open(TRANSCRIPT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_line)
    except Exception as e:
        log(f"Error writing transcript: {e}")


async def transcribe_audio_sarvam(audio_data: bytes, sample_rate: int = 16000) -> str:
    """Transcribe audio using Sarvam AI STT to see what user is saying"""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        return "[No Sarvam API key]"

    try:
        # Convert PCM to WAV format (Sarvam needs WAV)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        wav_buffer.seek(0)
        wav_bytes = wav_buffer.read()

        # Call Sarvam STT API
        url = "https://api.sarvam.ai/speech-to-text"

        form_data = aiohttp.FormData()
        form_data.add_field('file', wav_bytes, filename='audio.wav', content_type='audio/wav')
        form_data.add_field('model', 'saarika:v2.5')
        form_data.add_field('language_code', 'en-IN')  # English (India)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                data=form_data,
                headers={"api-subscription-key": api_key},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    transcript = result.get("transcript", "")
                    return transcript if transcript else "[silence]"
                else:
                    error_text = await response.text()
                    return f"[STT error: {response.status}]"
    except asyncio.TimeoutError:
        return "[STT timeout]"
    except Exception as e:
        return f"[STT error: {str(e)[:50]}]"

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    log("Loaded .env")
except ImportError:
    pass

# Google GenAI imports
try:
    from google import genai
    from google.genai import types
    log("Google GenAI SDK imported")
except ImportError as e:
    log(f"Google GenAI import error: {e}")
    log("Install with: pip install google-genai")
    sys.exit(1)

# WebSocket
import websockets

# numpy for audio
import numpy as np

log("Imports successful")


def load_system_prompt():
    """Load Freedom with AI sales script"""
    return """You are Aisha from Freedom with AI.

MOST IMPORTANT RULE: Actually LISTEN and RESPOND to what the user says. Do NOT follow a script.

EXAMPLES:
- User: "Can you hear me?" → You: "Yes, I can hear you clearly!"
- User: "Who is this?" → You: "This is Aisha from Freedom with AI."
- User: "What do you want?" → You: "I'm calling about the AI Masterclass you attended."
- User: "I'm busy" → You: "No problem, when should I call back?"
- User: "It was good" → You: "Great to hear! What did you like most about it?"
- User: "I didn't like it" → You: "I understand. What didn't work for you?"

RULES:
1. ALWAYS answer the user's actual question first
2. Keep responses SHORT - one sentence max
3. Be natural and conversational
4. If you don't understand, ask them to repeat

You're calling about the AI Masterclass they attended. Your goal is to have a natural conversation and understand their experience."""


class GeminiLiveSession:
    """
    Native Gemini Live audio streaming session
    Audio in → Gemini Live → Audio out
    """

    def __init__(self, call_id: str, caller_name: str, websocket):
        self.call_id = call_id
        self.caller_name = caller_name
        self.websocket = websocket

        self.client = None
        self.session = None
        self._session_context = None
        self._running = False

        # Tasks
        self._receive_task = None
        self._audio_send_task = None

        # Audio queue for sending to Gemini
        self._audio_queue = asyncio.Queue()

        # Counters for logging
        self._audio_in_count = 0
        self._audio_out_count = 0

    async def start(self) -> bool:
        """Start the Gemini Live session"""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            log(f"[{self.call_id}] GOOGLE_API_KEY not found")
            return False

        try:
            # Create client
            self.client = genai.Client(api_key=google_api_key)
            log(f"[{self.call_id}] Created GenAI client")

            # System prompt
            system_prompt = load_system_prompt()

            # Configure for native audio streaming
            # Note: Gemini Live only supports AUDIO modality for voice conversations
            config = types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Charon"  # Male voice - authoritative, professional
                        )
                    )
                ),
                system_instruction=types.Content(
                    parts=[types.Part(text=system_prompt)]
                ),
            )

            # Connect to Gemini Live with native audio model
            log(f"[{self.call_id}] Connecting to Gemini Live...")
            self.session = self.client.aio.live.connect(
                model="gemini-2.0-flash-exp",
                config=config
            )

            self._session_context = await self.session.__aenter__()
            log(f"[{self.call_id}] Connected to Gemini Live!")

            self._running = True

            # Start tasks
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._audio_send_task = asyncio.create_task(self._audio_send_loop())

            # Send greeting
            await self._send_greeting()

            return True

        except Exception as e:
            log(f"[{self.call_id}] Error starting session: {e}")
            import traceback
            log(traceback.format_exc())
            return False

    async def _send_greeting(self):
        """Send initial greeting"""
        try:
            await asyncio.sleep(0.5)

            log(f"[{self.call_id}] Sending greeting for {self.caller_name}...")

            # Simple greeting instruction
            await self._session_context.send_client_content(
                turns=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=f"The customer's name is {self.caller_name}. Start by saying: 'Hey {self.caller_name}, this is Aisha from Freedom with AI. You attended our AI Masterclass. Is this a good time to talk?' Then wait for their response.")]
                    )
                ],
                turn_complete=True
            )

            log(f"[{self.call_id}] Greeting sent!")

        except Exception as e:
            log(f"[{self.call_id}] Error sending greeting: {e}")

    async def push_audio(self, audio_data: bytes):
        """Push audio from WhatsApp to be sent to Gemini"""
        if self._running:
            self._audio_in_count += 1

            # Audio already amplified by main server (webrtc_handler)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            max_val = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0

            await self._audio_queue.put(audio_data)

            # Clear level-based logging
            if self._audio_in_count <= 10 or self._audio_in_count % 100 == 0:
                is_speech = "🗣️ SPEECH" if max_val > 3000 else "silence"
                log(f"📥 [WHATSAPP → SERVICE] #{self._audio_in_count}: {len(audio_data)}b, level={max_val:.0f} ({is_speech})")

    async def _audio_send_loop(self):
        """Send audio to Gemini Live"""
        log(f"[{self.call_id}] Audio send loop started")

        # Buffer to accumulate audio before sending
        audio_buffer = bytes()
        CHUNK_SIZE = 3200  # 100ms at 16kHz (16000 * 0.1 * 2 bytes) - smaller for faster response

        # Silence detection for turn signaling
        silence_threshold = 2000  # Audio level below this is silence (lowered for better detection)
        silence_start_time = None
        SILENCE_DURATION_FOR_TURN_END = 0.8  # Seconds of silence to trigger transcription (reduced)

        # Transcription buffer - accumulate speech to transcribe
        transcription_buffer = bytes()
        is_speaking = False
        MIN_SPEECH_BYTES = 8000  # Minimum 0.25 seconds of audio to transcribe (reduced)

        # Cooldown for transcription to prevent spamming
        last_transcription_time = 0
        TRANSCRIPTION_COOLDOWN = 2.0  # Minimum seconds between transcriptions

        while self._running:
            try:
                # Get audio from queue
                try:
                    audio_data = await asyncio.wait_for(self._audio_queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    # Check if we have buffered audio to send
                    if len(audio_buffer) >= CHUNK_SIZE:
                        await self._send_audio_to_gemini(audio_buffer)
                        audio_buffer = bytes()
                    continue

                if audio_data is None:
                    break

                # Add to buffer
                audio_buffer += audio_data

                # Check audio level for transcription logging
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                max_level = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0

                if max_level < silence_threshold:
                    # Silence detected
                    if silence_start_time is None:
                        silence_start_time = asyncio.get_event_loop().time()
                    else:
                        current_time = asyncio.get_event_loop().time()
                        silence_duration = current_time - silence_start_time

                        # When silence detected after speech, transcribe what user said
                        # Note: Gemini handles turn-taking automatically, no need to send activity_end
                        if silence_duration >= SILENCE_DURATION_FOR_TURN_END and is_speaking and len(transcription_buffer) >= MIN_SPEECH_BYTES:
                            # Check cooldown to prevent spamming transcriptions
                            if current_time - last_transcription_time >= TRANSCRIPTION_COOLDOWN:
                                is_speaking = False
                                last_transcription_time = current_time

                                # Transcribe in background to see what user said
                                log(f"🛑 [USER FINISHED] Detected {silence_duration:.1f}s silence - transcribing speech")
                                asyncio.create_task(self._transcribe_and_log(transcription_buffer))
                                transcription_buffer = bytes()
                else:
                    # Voice detected - reset silence timer
                    silence_start_time = None
                    is_speaking = True
                    transcription_buffer += audio_data  # Add to transcription buffer

                # Send when buffer is large enough
                if len(audio_buffer) >= CHUNK_SIZE:
                    await self._send_audio_to_gemini(audio_buffer)
                    audio_buffer = bytes()

            except asyncio.CancelledError:
                break
            except Exception as e:
                log(f"[{self.call_id}] Audio send error: {e}")
                import traceback
                log(traceback.format_exc())
                await asyncio.sleep(0.1)

        log(f"[{self.call_id}] Audio send loop ended")

    async def _transcribe_and_log(self, audio_data: bytes):
        """Transcribe audio and log what user said - GEMINI_IN level"""
        try:
            transcript = await transcribe_audio_sarvam(audio_data, sample_rate=16000)
            if transcript and not transcript.startswith("["):
                # Log to unified per-call flow file
                log_call_flow(self.call_id, "GEMINI_IN", "USER", transcript)
                # Also log to legacy transcript file
                log_transcript(self.call_id, "Customer", transcript)
            else:
                log(f"[STT Result] {transcript}")
        except Exception as e:
            log(f"[Transcription error] {e}")

    async def _send_audio_to_gemini(self, audio_data: bytes):
        """Send audio chunk to Gemini Live"""
        try:
            # Check if session is still valid
            if not self._session_context or not self._running:
                return

            # Track sends for logging
            if not hasattr(self, '_gemini_send_count'):
                self._gemini_send_count = 0
            self._gemini_send_count += 1

            # Calculate audio level
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            max_val = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0

            # Log audio being sent to Gemini
            if self._gemini_send_count <= 20 or self._gemini_send_count % 50 == 0:
                is_speech = "🗣️ SPEECH" if max_val > 3000 else "silence"
                log(f"🚀 [SERVICE → GEMINI] #{self._gemini_send_count}: {len(audio_data)}b, level={max_val:.0f} ({is_speech})")

            # Send raw PCM bytes to Gemini Live
            # Format: 16-bit signed PCM, mono, 16kHz
            await self._session_context.send_realtime_input(
                audio=types.Blob(
                    data=audio_data,  # Raw bytes (NOT base64)
                    mime_type="audio/pcm;rate=16000"
                )
            )
        except websockets.exceptions.ConnectionClosedError as e:
            log(f"[{self.call_id}] Connection closed while sending audio: {e}")
            # The receive loop will handle reconnection
        except Exception as e:
            log(f"[{self.call_id}] Error sending audio to Gemini: {e}")
            import traceback
            log(traceback.format_exc())

    async def _transcribe_agent_audio(self, audio_data: bytes):
        """Transcribe agent's audio response - GEMINI_OUT level"""
        try:
            # Agent audio is 24kHz, need to resample to 16kHz for Sarvam
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Resample 24kHz -> 16kHz (multiply by 2, divide by 3)
            from scipy.signal import resample_poly
            resampled = resample_poly(audio_array, 2, 3).astype(np.int16)

            transcript = await transcribe_audio_sarvam(resampled.tobytes(), sample_rate=16000)
            if transcript and not transcript.startswith("["):
                # Log to unified per-call flow file
                log_call_flow(self.call_id, "GEMINI_OUT", "AGENT", transcript)
                # Also log to legacy transcript file
                log_transcript(self.call_id, "AGENT", transcript)
        except Exception as e:
            log(f"[Agent transcription error] {e}")

    async def _receive_loop(self):
        """Receive responses from Gemini Live"""
        log(f"[{self.call_id}] Receive loop started")

        reconnect_attempts = 0
        max_reconnect_attempts = 3

        # Buffer for agent audio transcription
        agent_audio_buffer = bytes()
        last_activity_time = asyncio.get_event_loop().time()

        while self._running:
            try:
                async for response in self._session_context.receive():
                    if not self._running:
                        break

                    # Track activity for health monitoring
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_activity_time > 60:
                        log(f"⏱️ [HEALTH] Session active for {int(current_time - last_activity_time)}s")
                    last_activity_time = current_time

                    # Reset reconnect counter on successful receive
                    reconnect_attempts = 0

                    # Log all response types for debugging
                    if hasattr(response, 'setup_complete') and response.setup_complete:
                        log(f"[{self.call_id}] Setup complete")

                    if hasattr(response, 'tool_call') and response.tool_call:
                        log(f"[{self.call_id}] Tool call received")

                    # Handle server content (audio/text responses)
                    if response.server_content:
                        content = response.server_content

                        if content.turn_complete:
                            log(f"✅ [TURN COMPLETE] Ready for next user input")
                            # Transcribe what agent said when turn completes
                            if len(agent_audio_buffer) > 4800:  # At least 100ms of audio
                                asyncio.create_task(self._transcribe_agent_audio(agent_audio_buffer))
                            agent_audio_buffer = bytes()

                        if content.interrupted:
                            log(f"⚡ [INTERRUPTED] User interrupted agent response")
                            agent_audio_buffer = bytes()  # Clear buffer on interrupt

                        if content.model_turn and content.model_turn.parts:
                            for part in content.model_turn.parts:
                                # Log text responses from Gemini
                                if part.text:
                                    # Log to unified per-call flow file
                                    log_call_flow(self.call_id, "GEMINI_OUT", "AGENT", part.text)
                                    # Also log to legacy transcript file
                                    log_transcript(self.call_id, "AGENT", part.text)

                                # Send audio responses to WhatsApp
                                if part.inline_data:
                                    audio_data = part.inline_data.data
                                    mime_type = part.inline_data.mime_type

                                    # Handle if data is bytes or already decoded
                                    if isinstance(audio_data, str):
                                        audio_bytes = base64.b64decode(audio_data)
                                    else:
                                        audio_bytes = audio_data

                                    # Buffer agent audio for transcription
                                    agent_audio_buffer += audio_bytes

                                    self._audio_out_count += 1
                                    if self._audio_out_count <= 10 or self._audio_out_count % 50 == 0:
                                        log(f"📤 [GEMINI → SERVICE] #{self._audio_out_count}: {len(audio_bytes)}b audio response")

                                    await self._send_audio_to_whatsapp(audio_bytes)

                                    if self._audio_out_count <= 10 or self._audio_out_count % 50 == 0:
                                        log(f"📤 [SERVICE → WHATSAPP] #{self._audio_out_count}: Sent {len(audio_bytes)}b to caller")

            except asyncio.CancelledError:
                break
            except websockets.exceptions.ConnectionClosedError as e:
                if self._running:
                    reconnect_attempts += 1
                    log(f"[{self.call_id}] Gemini connection closed: {e}")

                    if reconnect_attempts <= max_reconnect_attempts:
                        log(f"[{self.call_id}] Attempting to reconnect ({reconnect_attempts}/{max_reconnect_attempts})...")

                        # Try to reconnect
                        try:
                            await self._reconnect()
                            log(f"[{self.call_id}] Reconnected successfully!")
                            continue  # Resume receive loop
                        except Exception as re:
                            log(f"[{self.call_id}] Reconnect failed: {re}")

                    if reconnect_attempts > max_reconnect_attempts:
                        log(f"[{self.call_id}] Max reconnect attempts reached, stopping session")
                        self._running = False
                        break
                else:
                    break
            except Exception as e:
                if self._running:
                    log(f"[{self.call_id}] Receive loop error: {e}")
                    import traceback
                    log(traceback.format_exc())
                    await asyncio.sleep(0.5)
                else:
                    break

        log(f"[{self.call_id}] Receive loop ended")

    async def _reconnect(self):
        """Attempt to reconnect to Gemini Live"""
        # Close existing session if any
        if self._session_context:
            try:
                await self.session.__aexit__(None, None, None)
            except:
                pass

        # Small delay before reconnecting
        await asyncio.sleep(1)

        # Recreate connection
        google_api_key = os.getenv("GOOGLE_API_KEY")
        system_prompt = load_system_prompt()

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Charon"
                    )
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=system_prompt)]
            ),
        )

        self.session = self.client.aio.live.connect(
            model="gemini-2.0-flash-exp",
            config=config
        )

        self._session_context = await self.session.__aenter__()
        log(f"[{self.call_id}] Reconnected to Gemini Live")

    async def _send_audio_to_whatsapp(self, audio_data: bytes):
        """Send audio to WhatsApp via WebSocket"""
        try:
            # Handle base64 encoded data from Gemini
            if isinstance(audio_data, str):
                audio_data = base64.b64decode(audio_data)

            audio_hex = audio_data.hex()
            await self.websocket.send(json.dumps({
                "type": "audio",
                "data": audio_hex,
                "sample_rate": 24000,  # Gemini outputs 24kHz
                "num_channels": 1
            }))
        except Exception as e:
            log(f"[{self.call_id}] Error sending audio to WhatsApp: {e}")

    async def stop(self):
        """Stop the session"""
        log(f"[{self.call_id}] Stopping session...")
        self._running = False

        # Signal audio loop to stop
        await self._audio_queue.put(None)

        # Cancel tasks
        for task in [self._receive_task, self._audio_send_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close Gemini session
        if self._session_context:
            try:
                await self.session.__aexit__(None, None, None)
            except:
                pass

        log(f"[{self.call_id}] Session stopped")


# Global sessions
active_sessions = {}


async def handle_websocket(websocket):
    """Handle WebSocket connection from main server"""
    call_id = None
    session = None

    try:
        # Only log real websocket connections (not health checks)
        remote = websocket.remote_address
        if remote:
            log(f"WebSocket connection from {remote}")

        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "start":
                    call_id = data.get("call_id")
                    caller_name = data.get("caller_name", "Customer")

                    log(f"[{call_id}] Starting Gemini Live session for {caller_name}")

                    session = GeminiLiveSession(call_id, caller_name, websocket)

                    if await session.start():
                        active_sessions[call_id] = session
                        await websocket.send(json.dumps({
                            "type": "started",
                            "call_id": call_id
                        }))
                        log(f"[{call_id}] Session started successfully")
                    else:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "Failed to create Gemini Live session"
                        }))

                elif msg_type == "audio":
                    if session and call_id in active_sessions:
                        audio_hex = data.get("data", "")
                        if audio_hex:
                            audio_data = bytes.fromhex(audio_hex)
                            await session.push_audio(audio_data)

                elif msg_type == "stop":
                    if session and call_id in active_sessions:
                        await session.stop()
                        del active_sessions[call_id]
                        await websocket.send(json.dumps({
                            "type": "stopped",
                            "call_id": call_id
                        }))
                        log(f"[{call_id}] Session stopped by request")

            except json.JSONDecodeError:
                log("Invalid JSON received")
            except Exception as e:
                log(f"Error handling message: {e}")
                import traceback
                traceback.print_exc()

    except websockets.exceptions.ConnectionClosedOK:
        # Normal close - don't log as error
        if call_id:
            log(f"[{call_id}] WebSocket closed normally")
    except websockets.exceptions.ConnectionClosedError as e:
        if call_id:
            log(f"[{call_id}] WebSocket closed with error: {e}")
    except websockets.exceptions.ConnectionClosed:
        if call_id:
            log(f"[{call_id}] WebSocket closed")
    except Exception as e:
        # Catch any other exceptions (including handshake failures)
        if call_id:
            log(f"[{call_id}] WebSocket error: {e}")
        # Don't log errors for health check / scanner connections
    finally:
        if session and call_id in active_sessions:
            await session.stop()
            del active_sessions[call_id]


async def main():
    """Start server"""
    port = int(os.getenv("GEMINI_LIVE_PORT", 8003))

    log("=" * 60)
    log("Gemini Live Service (Native Audio Streaming)")
    log("Audio In → Gemini Live → Audio Out")
    log("No separate STT/LLM/TTS - all handled by Gemini")
    log("=" * 60)
    log(f"Starting on ws://0.0.0.0:{port}")

    # Configure websocket server
    async with websockets.serve(
        handle_websocket,
        "0.0.0.0",
        port,
        ping_interval=30,
        ping_timeout=10,
    ):
        log(f"Server running on ws://0.0.0.0:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Server stopped")
