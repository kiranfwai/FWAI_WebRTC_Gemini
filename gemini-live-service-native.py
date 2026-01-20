"""
Gemini Live Service for WhatsApp Calls
Uses Google GenAI SDK directly for native audio streaming
"""

import os
import asyncio
import json
import sys
import base64
from datetime import datetime

# Setup logging
LOG_FILE = "gemini-live-service.log"

def log_to_file(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

with open(LOG_FILE, "w") as f:
    f.write(f"=== Gemini Live Service Log - Started at {datetime.now()} ===\n\n")

def log(message):
    print(message)
    log_to_file(message)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    log("Loaded environment variables from .env file")
except ImportError:
    log("python-dotenv not installed")

# Google GenAI imports
try:
    from google import genai
    from google.genai import types
    log("Google GenAI SDK imported successfully")
except ImportError as e:
    log(f"Google GenAI import error: {e}")
    sys.exit(1)

# WebSocket imports
try:
    import websockets
    from websockets.server import serve
except ImportError:
    log("websockets not installed")
    sys.exit(1)

# numpy for audio processing
try:
    import numpy as np
except ImportError:
    log("numpy not installed")
    sys.exit(1)


def load_conversation_script():
    """Load conversation script for the AI agent"""
    return """You are Mousumi, a Senior AI Counselor at Freedom with AI.

VOICE & PERSONALITY:
- Indian English accent, natural and warm
- Professional yet friendly, like a trusted career advisor
- Empathetic listener who validates concerns
- Confident about AI education and career opportunities

CONVERSATION APPROACH (NEPQ Sales Method):
1. CONNECT: Build rapport, understand why they attended the masterclass
2. ENGAGE: Explore their current situation with AI/career
3. PROBLEM AWARE: Uncover challenges they face
4. SOLUTION AWARE: Discuss what they've tried, what they need
5. CONSEQUENCE: Help them see impact of not acting
6. PRESENT: Share Freedom with AI Gold Membership benefits
7. COMMIT: Guide toward enrollment

KEY TALKING POINTS:
- You attended our AI Masterclass with Avinash recently
- We help people advance careers and increase income with AI skills
- Gold Membership: 14,000 INR + GST for 12 months includes:
  * Comprehensive AI & Prompt Engineering courses
  * Exclusive WhatsApp community for networking
  * Live Inner Circle calls with expert mentors (Avinash, Madhu, Leela)
  * 500+ prompt library for real-world applications

RESPONSE RULES:
- Ask ONE question at a time, then STOP and listen
- Keep responses concise (2-3 sentences max)
- Respond quickly and naturally
- Acknowledge what they say before asking next question
- If they seem interested, guide toward enrollment
- Handle objections with empathy (cost, time, skepticism)

When SYSTEM_COMMAND is given, speak the greeting immediately."""


class GeminiLiveSession:
    """
    Manages a Gemini Live session using the native Google GenAI SDK
    Handles bidirectional audio streaming
    """
    
    def __init__(self, call_id: str, caller_name: str, websocket):
        self.call_id = call_id
        self.caller_name = caller_name
        self.websocket = websocket
        self.client = None
        self.session = None
        self._running = False
        self._audio_queue = asyncio.Queue()
        self._send_task = None
        self._receive_task = None
        self._audio_send_count = 0
        self._audio_recv_count = 0
        self._audio_buffer = []  # Buffer to record incoming audio for debugging
        
    async def start(self):
        """Start the Gemini Live session"""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            log(f"[{self.call_id}] GOOGLE_API_KEY not found")
            return False
            
        try:
            # Create client
            self.client = genai.Client(api_key=google_api_key)
            log(f"[{self.call_id}] Created GenAI client")
            
            # Configure the session
            system_prompt = load_conversation_script()
            
            config = types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Charon"  # Indian-sounding voice
                        )
                    )
                ),
                system_instruction=types.Content(
                    parts=[types.Part(text=system_prompt)]
                ),
            )
            
            # Connect to Gemini Live - use Gemini 2.5 Flash Native Audio
            log(f"[{self.call_id}] Connecting to Gemini Live...")
            self.session = self.client.aio.live.connect(
                model="models/gemini-2.5-flash-native-audio-preview-12-2025",
                config=config
            )
            
            # Enter the session context
            self._session_context = await self.session.__aenter__()
            log(f"[{self.call_id}] Connected to Gemini Live!")
            
            self._running = True
            
            # Start receive task
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            # Start send task
            self._send_task = asyncio.create_task(self._send_loop())
            
            # Trigger greeting
            await self._trigger_greeting()
            
            return True
            
        except Exception as e:
            log(f"[{self.call_id}] Error starting session: {e}")
            import traceback
            log(traceback.format_exc())
            return False
    
    async def _trigger_greeting(self):
        """Trigger the initial greeting"""
        try:
            await asyncio.sleep(0.5)  # Wait for connection to stabilize
            
            user_ref = f"Hi {self.caller_name}" if self.caller_name and self.caller_name != "Unknown" else "Hi"
            greeting_text = f"{user_ref}, this is Mousumi from Freedom with AI. I noticed you showed interest in our AI programs. How are you doing today?"
            
            trigger_msg = f"SYSTEM_COMMAND: Speak this EXACT greeting immediately with Indian Accent: '{greeting_text}'"
            
            log(f"[{self.call_id}] Sending greeting trigger...")
            await self._session_context.send(input=trigger_msg, end_of_turn=True)
            log(f"[{self.call_id}] Greeting trigger sent!")
            
        except Exception as e:
            log(f"[{self.call_id}] Error triggering greeting: {e}")
    
    async def push_audio(self, audio_data: bytes):
        """Push audio data from WhatsApp to be sent to Gemini"""
        if self._running:
            # Analyze and amplify audio
            try:
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                
                if len(audio_array) > 0:
                    max_val = np.max(np.abs(audio_array))
                    
                    self._audio_send_count += 1
                    
                    # Log audio levels
                    if self._audio_send_count <= 10 or self._audio_send_count % 100 == 0:
                        rms = np.sqrt(np.mean(audio_array ** 2))
                        log(f"[{self.call_id}] [AUDIO_IN] #{self._audio_send_count}: max={max_val:.0f}, rms={rms:.0f}")
                    
                    # Amplify low-level audio
                    if max_val > 1 and max_val < 10000:
                        target_level = 14000
                        gain = target_level / max_val
                        if max_val < 500:
                            gain = min(gain, 20.0)
                        elif max_val < 2000:
                            gain = min(gain, 10.0)
                        else:
                            gain = min(gain, 5.0)
                        
                        audio_array = audio_array * gain
                        audio_array = np.clip(audio_array, -32768, 32767).astype(np.int16)
                        audio_data = audio_array.tobytes()
                        
                        if self._audio_send_count <= 5:
                            log(f"[{self.call_id}] [AMPLIFY] gain={gain:.1f}x, new_max={np.max(np.abs(audio_array)):.0f}")
                            
            except Exception as e:
                if self._audio_send_count <= 3:
                    log(f"[{self.call_id}] Audio processing error: {e}")
            
            # Save audio to buffer for debugging/transcription
            self._audio_buffer.append(audio_data)
            
            await self._audio_queue.put(audio_data)
    
    async def _send_loop(self):
        """Send audio to Gemini Live"""
        log(f"[{self.call_id}] Audio send loop started")
        
        # Buffer to accumulate audio chunks
        audio_buffer = bytes()
        CHUNK_SIZE = 4096  # Send in larger chunks for efficiency
        
        while self._running:
            try:
                # Get audio with timeout
                try:
                    audio_data = await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    # Send any buffered audio
                    if len(audio_buffer) > 0:
                        await self._send_audio_chunk(audio_buffer)
                        audio_buffer = bytes()
                    continue
                
                if audio_data is None:
                    break
                
                # Add to buffer
                audio_buffer += audio_data
                
                # Send when buffer is large enough
                if len(audio_buffer) >= CHUNK_SIZE:
                    await self._send_audio_chunk(audio_buffer)
                    audio_buffer = bytes()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                log(f"[{self.call_id}] Send loop error: {e}")
                await asyncio.sleep(0.1)
        
        log(f"[{self.call_id}] Audio send loop ended")
    
    async def _send_audio_chunk(self, audio_data: bytes):
        """Send a chunk of audio to Gemini"""
        try:
            # Gemini Live expects raw audio as blob
            audio_blob = types.Blob(
                mime_type="audio/pcm;rate=16000",
                data=audio_data
            )
            
            await self._session_context.send(input=audio_blob)
            
            # Simple VAD: track if we're in speech or silence
            if not hasattr(self, '_in_speech'):
                self._in_speech = False
                self._silence_count = 0
                self._speech_frames = 0
            
            # Analyze audio level
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            max_val = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0
            
            if max_val > 1000:  # Speech detected
                self._in_speech = True
                self._speech_frames += 1
                self._silence_count = 0
            elif self._in_speech:
                self._silence_count += 1
                # If we had speech and now 20+ frames of silence (~0.5 sec), log it
                if self._silence_count >= 20 and self._speech_frames >= 10:
                    log(f"[{self.call_id}] Detected end of speech (speech_frames={self._speech_frames})")
                    # Note: Gemini Live should use its built-in VAD to detect turn end
                    # Manual end_of_turn signaling causes errors
                    self._in_speech = False
                    self._speech_frames = 0
                    self._silence_count = 0
            
        except Exception as e:
            if hasattr(self, '_audio_send_count') and self._audio_send_count <= 5:
                log(f"[{self.call_id}] Error sending audio: {e}")
    
    async def _receive_loop(self):
        """Receive responses from Gemini Live - keep running continuously"""
        log(f"[{self.call_id}] Receive loop started")
        
        while self._running:
            try:
                async for response in self._session_context.receive():
                    if not self._running:
                        break
                        
                    # Handle server content (audio/text responses)
                    if response.server_content:
                        content = response.server_content
                        
                        # Check if model is done speaking (turn complete)
                        if content.turn_complete:
                            log(f"[{self.call_id}] Turn complete - waiting for more input...")
                            # Don't break - keep listening for more
                        
                        # Log input transcription (what user said)
                        if hasattr(content, 'input_transcription') and content.input_transcription:
                            log(f"[{self.call_id}] ★★★ USER SAID: \"{content.input_transcription}\" ★★★")
                        
                        # Log output transcription (what bot said in text)
                        if hasattr(content, 'output_transcription') and content.output_transcription:
                            log(f"[{self.call_id}] ★★★ BOT TEXT: \"{content.output_transcription}\" ★★★")
                        
                        # Process model parts
                        if content.model_turn and content.model_turn.parts:
                            for part in content.model_turn.parts:
                                # Handle text
                                if part.text:
                                    log(f"[{self.call_id}] ★★★ BOT SAID: \"{part.text}\" ★★★")
                                
                                # Handle audio
                                if part.inline_data:
                                    audio_data = part.inline_data.data
                                    self._audio_recv_count += 1
                                    
                                    if self._audio_recv_count <= 5 or self._audio_recv_count % 50 == 0:
                                        log(f"[{self.call_id}] [AUDIO_OUT] #{self._audio_recv_count}: {len(audio_data)} bytes")
                                    
                                    # Send audio to WhatsApp via WebSocket
                                    await self._send_audio_to_whatsapp(audio_data)
                    
                    # Handle tool calls if any
                    if response.tool_call:
                        log(f"[{self.call_id}] Tool call received (not implemented)")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    log(f"[{self.call_id}] Receive loop error: {e}")
                    import traceback
                    log(traceback.format_exc())
                    await asyncio.sleep(0.5)  # Wait before retrying
                else:
                    break
        
        log(f"[{self.call_id}] Receive loop ended")
    
    async def _send_audio_to_whatsapp(self, audio_data: bytes):
        """Send audio response back to WhatsApp"""
        try:
            # Gemini outputs 24kHz audio
            audio_hex = audio_data.hex() if isinstance(audio_data, bytes) else bytes(audio_data).hex()
            
            await self.websocket.send(json.dumps({
                "type": "audio",
                "data": audio_hex,
                "sample_rate": 24000,
                "num_channels": 1
            }))
            
        except Exception as e:
            log(f"[{self.call_id}] Error sending audio to WhatsApp: {e}")
    
    async def stop(self):
        """Stop the session"""
        log(f"[{self.call_id}] Stopping session...")
        self._running = False
        
        # Signal send loop to stop
        await self._audio_queue.put(None)
        
        # Cancel tasks
        if self._send_task:
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        # Close session
        if self._session_context:
            try:
                await self.session.__aexit__(None, None, None)
            except:
                pass
        
        # Save and transcribe recorded audio
        await self._save_and_transcribe_audio()
        
        log(f"[{self.call_id}] Session stopped")
    
    async def _save_and_transcribe_audio(self):
        """Save recorded audio and transcribe it"""
        if not self._audio_buffer:
            log(f"[{self.call_id}] No audio recorded")
            return
            
        try:
            # Combine all audio chunks
            all_audio = b''.join(self._audio_buffer)
            log(f"[{self.call_id}] Recorded {len(all_audio)} bytes of audio ({len(self._audio_buffer)} chunks)")
            
            # Save as WAV file
            import wave
            wav_path = f"/app/call_{self.call_id}.wav"
            with wave.open(wav_path, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(16000)  # 16kHz
                wav.writeframes(all_audio)
            log(f"[{self.call_id}] Saved audio to {wav_path}")
            
            # Transcribe using Gemini
            await self._transcribe_audio(all_audio)
            
        except Exception as e:
            log(f"[{self.call_id}] Error saving audio: {e}")
    
    async def _transcribe_audio(self, audio_data: bytes):
        """Transcribe audio using Gemini"""
        try:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                return
            
            client = genai.Client(api_key=google_api_key)
            
            # Use Gemini to transcribe
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    types.Part.from_bytes(
                        data=audio_data,
                        mime_type="audio/wav"
                    ),
                    "Transcribe this audio. Just output the text spoken, nothing else."
                ]
            )
            
            if response.text:
                log(f"[{self.call_id}] ★★★ TRANSCRIPTION OF USER SPEECH: \"{response.text}\" ★★★")
            else:
                log(f"[{self.call_id}] No transcription returned")
                
        except Exception as e:
            log(f"[{self.call_id}] Transcription error: {e}")


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
                    caller_name = data.get("caller_name", "Unknown")
                    
                    if call_id:
                        log(f"[{call_id}] Starting session for {caller_name}")
                        
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
                                "message": "Failed to create session"
                            }))
                    else:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "call_id required"
                        }))
                
                elif msg_type == "audio":
                    # Receive audio from WhatsApp
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
                log(f"Invalid JSON received")
            except Exception as e:
                log(f"Error handling message: {e}")
                import traceback
                log(traceback.format_exc())
    
    except websockets.exceptions.ConnectionClosed:
        log(f"WebSocket closed for call {call_id}")
    except Exception as e:
        log(f"WebSocket error: {e}")
    finally:
        if session and call_id in active_sessions:
            await session.stop()
            del active_sessions[call_id]


async def main():
    """Start WebSocket server"""
    port = int(os.getenv("GEMINI_LIVE_PORT", 8003))
    log("=" * 60)
    log("Gemini Live Service (Native Google GenAI SDK)")
    log("=" * 60)
    log(f"Starting on ws://0.0.0.0:{port}")
    
    async with serve(handle_websocket, "0.0.0.0", port):
        log(f"Gemini Live Service running on ws://0.0.0.0:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Server stopped")
