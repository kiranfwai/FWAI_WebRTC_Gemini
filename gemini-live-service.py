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

# Setup logging
def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


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
    return """You are an AI Sales Strategist at Freedom with AI. Your name is Aisha.

VOICE & STYLE:
- Speak naturally with a warm, professional Indian English tone
- Keep responses SHORT - maximum 2-3 sentences
- Ask ONE question at a time, then WAIT for response
- Be conversational, not scripted

CONTEXT:
- The caller attended Avinash's AI Masterclass recently
- You're following up to understand their goals and help them
- You represent Freedom with AI's membership program (14,000 INR + GST for 12 months)

CONVERSATION FLOW:
1. GREETING: "Hey! This is Aisha from Freedom with AI. I'm an AI Strategist here. I noticed you attended our AI Masterclass with Avinash. This call is to understand what you're looking for and see how we can help. Is that fine?"

2. CONNECTING: Ask what attracted them to the masterclass? What made them stay till the end?

3. SITUATION:
   - How long have they been using AI/ChatGPT?
   - What got them curious about AI?
   - What are they doing now to future-proof their skills?

4. PROBLEM AWARENESS:
   - Are they happy with their current approach to learning AI?
   - What challenges are they facing?
   - What would happen if they don't take action?

5. SOLUTION:
   - Have they looked for AI training programs before?
   - What are they looking for in training and support?
   - What's their goal - how much extra income do they want?

6. PRESENTATION (only when they're ready):
   "Our program is 14,000 plus GST for 12 months. You get:
   - Pillar 1: Comprehensive AI education with coding, prompt engineering, monetization strategies
   - Pillar 2: Exclusive WhatsApp community for networking
   - Pillar 3: Live mentorship calls with Avinash and other experts"

7. CLOSE: Ask if they feel this could be the answer for them? Guide toward enrollment.

RULES:
- LISTEN to what they actually say and respond naturally
- Don't be pushy - be helpful and consultative
- If they mention concerns, acknowledge and address them
- Keep it conversational, not like reading a script
- If they're not interested, respect that gracefully"""


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
            config = types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Charon"  # Good for Indian accent
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

            greeting = f"Hey {self.caller_name}! This is Aisha from Freedom with AI. I'm an AI Strategist here. I noticed you attended our AI Masterclass with Avinash recently. This call is just to understand what you're looking for and see how we can help. Is that fine with you?"

            log(f"[{self.call_id}] Sending greeting...")

            # Send text for Gemini to speak
            await self._session_context.send_client_content(
                turns=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=f"[System: Greet the caller] Say: {greeting}")]
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
            # Just log levels and pass through - no additional amplification to avoid distortion
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            max_val = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0

            await self._audio_queue.put(audio_data)

            if self._audio_in_count <= 10 or self._audio_in_count % 100 == 0:
                log(f"[{self.call_id}] [AUDIO_IN] #{self._audio_in_count}: {len(audio_data)} bytes, max_level={max_val:.0f}")

    async def _audio_send_loop(self):
        """Send audio to Gemini Live"""
        log(f"[{self.call_id}] Audio send loop started")

        # Buffer to accumulate audio before sending
        audio_buffer = bytes()
        CHUNK_SIZE = 3200  # 100ms at 16kHz (16000 * 0.1 * 2 bytes) - smaller for faster response

        # Silence detection for turn signaling
        silence_threshold = 3000  # Audio level below this is silence (after 800x amplification)
        silence_start_time = None
        SILENCE_DURATION_FOR_TURN_END = 1.5  # Seconds of silence to signal turn complete

        # Transcription buffer - accumulate speech to transcribe
        transcription_buffer = bytes()
        is_speaking = False
        MIN_SPEECH_BYTES = 16000  # Minimum 0.5 seconds of audio to transcribe

        # Cooldown for activity_end to prevent spamming
        last_activity_end_time = 0
        ACTIVITY_END_COOLDOWN = 3.0  # Minimum seconds between activity_end signals

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

                        # When silence detected after speech, signal end of turn to Gemini
                        if silence_duration >= SILENCE_DURATION_FOR_TURN_END and is_speaking and len(transcription_buffer) >= MIN_SPEECH_BYTES:
                            # Check cooldown to prevent spamming activity_end
                            if current_time - last_activity_end_time >= ACTIVITY_END_COOLDOWN:
                                is_speaking = False
                                last_activity_end_time = current_time

                                # Transcribe in background to see what user said
                                asyncio.create_task(self._transcribe_and_log(transcription_buffer))
                                transcription_buffer = bytes()

                                # Signal to Gemini that user has finished speaking
                                # This tells Gemini it's time to respond
                                try:
                                    log(f"[{self.call_id}] Signaling activity_end (user finished speaking, {silence_duration:.1f}s silence)")
                                    await self._session_context.send_realtime_input(
                                        activity_end=types.ActivityEnd()
                                    )
                                except Exception as ae:
                                    log(f"[{self.call_id}] Error sending activity_end: {ae}")
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
        """Transcribe audio and log what user said"""
        try:
            transcript = await transcribe_audio_sarvam(audio_data, sample_rate=16000)
            log(f"[{self.call_id}] *** USER SAID: \"{transcript}\" ***")
        except Exception as e:
            log(f"[{self.call_id}] Transcription error: {e}")

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

            # Log audio being sent - more frequent initially
            if self._gemini_send_count <= 20 or self._gemini_send_count % 50 == 0:
                is_speech = "SPEECH" if max_val > 500 else "silence"
                log(f"[{self.call_id}] [GEMINI_SEND] #{self._gemini_send_count}: {len(audio_data)} bytes, max={max_val:.0f} ({is_speech})")

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

    async def _receive_loop(self):
        """Receive responses from Gemini Live"""
        log(f"[{self.call_id}] Receive loop started")

        reconnect_attempts = 0
        max_reconnect_attempts = 3

        while self._running:
            try:
                async for response in self._session_context.receive():
                    if not self._running:
                        break

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
                            log(f"[{self.call_id}] Turn complete - ready for next input")

                        if content.interrupted:
                            log(f"[{self.call_id}] Response interrupted by user")

                        if content.model_turn and content.model_turn.parts:
                            for part in content.model_turn.parts:
                                # Log text responses
                                if part.text:
                                    log(f"[{self.call_id}] BOT TEXT: {part.text[:100]}...")

                                # Send audio responses to WhatsApp
                                if part.inline_data:
                                    audio_data = part.inline_data.data
                                    mime_type = part.inline_data.mime_type

                                    # Handle if data is bytes or already decoded
                                    if isinstance(audio_data, str):
                                        # It's base64 encoded string
                                        audio_bytes = base64.b64decode(audio_data)
                                    else:
                                        # It's already bytes
                                        audio_bytes = audio_data

                                    self._audio_out_count += 1
                                    if self._audio_out_count <= 10 or self._audio_out_count % 50 == 0:
                                        log(f"[{self.call_id}] [AUDIO_OUT] #{self._audio_out_count}: {len(audio_bytes)} bytes, mime={mime_type}")

                                    await self._send_audio_to_whatsapp(audio_bytes)

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
        log(f"WebSocket connection from {websocket.remote_address}")

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

    except websockets.exceptions.ConnectionClosed:
        log(f"WebSocket closed for call {call_id}")
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

    async with websockets.serve(handle_websocket, "0.0.0.0", port):
        log(f"Server running on ws://0.0.0.0:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Server stopped")
