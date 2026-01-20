"""
STT-Only Service - Just transcribes what user says
No Gemini AI - replaces gemini-live-service.py for testing
"""

import os
import asyncio
import json
import io
import wave
import aiohttp
import numpy as np
import websockets
from datetime import datetime

def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

# Log file for transcriptions
TRANSCRIPT_LOG_FILE = "/app/audio_debug/call_transcripts.log"

def log_transcript(call_id: str, caller_name: str, transcript: str):
    """Save transcript to log file"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(TRANSCRIPT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [{call_id[:8]}] {caller_name}: {transcript}\n")
    except Exception as e:
        log(f"Error writing transcript log: {e}")

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")


async def transcribe_audio_sarvam(audio_data: bytes, sample_rate: int = 16000) -> str:
    """Transcribe audio using Sarvam AI STT"""
    if not SARVAM_API_KEY:
        return "[No Sarvam API key]"

    try:
        # Limit audio to ~30 seconds max (480000 bytes at 16kHz)
        MAX_BYTES = 480000
        if len(audio_data) > MAX_BYTES:
            log(f"Audio too long ({len(audio_data)} bytes), truncating to {MAX_BYTES}")
            audio_data = audio_data[:MAX_BYTES]

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        wav_buffer.seek(0)
        wav_bytes = wav_buffer.read()

        log(f"Sending {len(wav_bytes)} bytes to Sarvam STT...")

        url = "https://api.sarvam.ai/speech-to-text"
        form_data = aiohttp.FormData()
        form_data.add_field('file', wav_bytes, filename='audio.wav', content_type='audio/wav')
        form_data.add_field('model', 'saarika:v2.5')
        form_data.add_field('language_code', 'en-IN')  # Change to 'hi-IN' for Hindi

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                data=form_data,
                headers={"api-subscription-key": SARVAM_API_KEY},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_text = await response.text()
                log(f"Sarvam response: {response.status} - {response_text[:200]}")

                if response.status == 200:
                    result = await response.json()
                    transcript = result.get("transcript", "")
                    return transcript if transcript else "[empty]"
                return f"[STT error {response.status}: {response_text[:100]}]"
    except asyncio.TimeoutError:
        return "[STT timeout]"
    except Exception as e:
        log(f"STT error: {e}")
        return f"[Error: {str(e)[:50]}]"


class SimpleSTTSession:
    """Just receives audio and transcribes - no AI response"""

    def __init__(self, call_id: str, caller_name: str, websocket):
        self.call_id = call_id
        self.caller_name = caller_name
        self.websocket = websocket
        self._running = False
        self._audio_queue = asyncio.Queue()
        self._task = None
        self._audio_count = 0

    async def start(self) -> bool:
        self._running = True
        self._task = asyncio.create_task(self._process_audio())
        log(f"[{self.call_id}] STT session started for {self.caller_name}")

        # Send a simple confirmation (no actual audio)
        await self.websocket.send(json.dumps({
            "type": "started",
            "call_id": self.call_id
        }))

        return True

    async def push_audio(self, audio_data: bytes):
        """Receive audio from WhatsApp"""
        if self._running:
            self._audio_count += 1

            # Check audio level - NO amplification here (already done in webrtc_handler)
            # Just pass through the audio as-is
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            max_val = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0

            await self._audio_queue.put(audio_data)

            # Log audio levels - more frequently for debugging
            if self._audio_count <= 20 or self._audio_count % 25 == 0:
                log(f"[{self.call_id}] Audio #{self._audio_count}: {len(audio_data)} bytes, max={max_val}")

    async def _process_audio(self):
        """Process incoming audio and transcribe"""
        log(f"[{self.call_id}] Audio processing started")

        # Save ALL audio to a single debug file
        all_audio_buffer = bytes()

        # Accumulate audio for transcription
        audio_buffer = bytes()
        silence_threshold = 100  # Lowered from 500 - audio is quiet even after amplification
        consecutive_silence_frames = 0
        FRAMES_FOR_SILENCE = 15  # ~0.6 seconds at 40ms/frame (lowered from 25)
        speech_detected = False
        transcription_count = 0

        while self._running:
            try:
                try:
                    audio_data = await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    # Check for timeout-based transcription
                    if speech_detected and len(audio_buffer) >= 16000 and consecutive_silence_frames >= FRAMES_FOR_SILENCE:
                        log(f"[{self.call_id}] Timeout transcription (buffer={len(audio_buffer)} bytes)")
                        speech_detected = False
                        transcription_count += 1
                        transcript = await transcribe_audio_sarvam(audio_buffer)
                        log(f"")
                        log(f"╔══════════════════════════════════════════════════════════════╗")
                        log(f"║ 🎤 USER SAID #{transcription_count}: \"{transcript}\"")
                        log(f"╚══════════════════════════════════════════════════════════════╝")
                        log(f"")
                        # Save to transcript log file
                        if transcript and transcript not in ["[empty]", "[No Sarvam API key]"]:
                            log_transcript(self.call_id, self.caller_name, transcript)
                        audio_buffer = bytes()
                        consecutive_silence_frames = 0
                    continue

                if audio_data is None:
                    break

                # Save ALL audio for complete debug file
                all_audio_buffer += audio_data

                # Check audio level
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                max_level = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0

                if max_level > silence_threshold:
                    # Speech detected
                    if not speech_detected:
                        log(f"[{self.call_id}] 🎙️ Speech started (level={max_level})")
                    speech_detected = True
                    consecutive_silence_frames = 0
                    audio_buffer += audio_data
                else:
                    # Silence frame
                    consecutive_silence_frames += 1

                    # If we were speaking and now have enough silence, transcribe
                    if speech_detected and len(audio_buffer) >= 16000 and consecutive_silence_frames >= FRAMES_FOR_SILENCE:
                        log(f"[{self.call_id}] Silence detected ({consecutive_silence_frames} frames), transcribing {len(audio_buffer)} bytes...")
                        speech_detected = False
                        transcription_count += 1

                        # Save audio to file for debugging
                        filename = f"/app/audio_debug/audio_{transcription_count}.wav"
                        try:
                            wav_buffer = io.BytesIO()
                            with wave.open(wav_buffer, 'wb') as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(16000)
                                wf.writeframes(audio_buffer)
                            with open(filename, 'wb') as f:
                                f.write(wav_buffer.getvalue())
                            log(f"[{self.call_id}] Saved audio to {filename}")
                        except Exception as e:
                            log(f"Error saving audio: {e}")

                        # Transcribe
                        transcript = await transcribe_audio_sarvam(audio_buffer)
                        log(f"")
                        log(f"╔══════════════════════════════════════════════════════════════╗")
                        log(f"║ 🎤 USER SAID #{transcription_count}: \"{transcript}\"")
                        log(f"║ 📁 Audio saved to: {filename}")
                        log(f"╚══════════════════════════════════════════════════════════════╝")
                        log(f"")

                        # Save to transcript log file
                        if transcript and transcript not in ["[empty]", "[No Sarvam API key]"]:
                            log_transcript(self.call_id, self.caller_name, transcript)

                        audio_buffer = bytes()
                        consecutive_silence_frames = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                log(f"[{self.call_id}] Error: {e}")
                await asyncio.sleep(0.1)

        # Save ALL audio to complete debug file when call ends
        if len(all_audio_buffer) > 0:
            try:
                # Convert to numpy for normalization
                audio_array = np.frombuffer(all_audio_buffer, dtype=np.int16).astype(np.float32)

                # Apply global normalization - find max and scale to 80% of max volume
                max_val = np.max(np.abs(audio_array))
                if max_val > 0:
                    target_level = 32767 * 0.8  # 80% of max to avoid clipping
                    gain = target_level / max_val
                    audio_normalized = (audio_array * gain).astype(np.int16)
                    normalized_buffer = audio_normalized.tobytes()
                    log(f"[{self.call_id}] Normalized audio: max {max_val:.0f} -> {target_level:.0f} (gain: {gain:.1f}x)")
                else:
                    normalized_buffer = all_audio_buffer

                # Save RAW (unnormalized) version
                raw_filename = f"/app/audio_debug/raw_call_{self.call_id[:8]}.wav"
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(all_audio_buffer)
                with open(raw_filename, 'wb') as f:
                    f.write(wav_buffer.getvalue())

                # Save NORMALIZED version
                complete_filename = f"/app/audio_debug/normalized_call_{self.call_id[:8]}.wav"
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(normalized_buffer)
                with open(complete_filename, 'wb') as f:
                    f.write(wav_buffer.getvalue())

                duration_sec = len(all_audio_buffer) / (16000 * 2)
                log(f"")
                log(f"╔══════════════════════════════════════════════════════════════╗")
                log(f"║ 📁 COMPLETE CALL AUDIO SAVED")
                log(f"║ Raw: {raw_filename}")
                log(f"║ Normalized: {complete_filename}")
                log(f"║ Duration: {duration_sec:.1f} seconds")
                log(f"╚══════════════════════════════════════════════════════════════╝")
                log(f"")
            except Exception as e:
                log(f"Error saving complete audio: {e}")

        log(f"[{self.call_id}] Audio processing ended")

    async def stop(self):
        log(f"[{self.call_id}] Stopping session...")
        self._running = False
        await self._audio_queue.put(None)  # Signal to exit the loop
        if self._task:
            # Wait for the task to finish naturally (it will save audio)
            # Give it up to 5 seconds to save the audio
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                log(f"[{self.call_id}] Timeout waiting for audio save, cancelling...")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
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

                    log(f"[{call_id}] Starting STT session for {caller_name}")

                    session = SimpleSTTSession(call_id, caller_name, websocket)
                    if await session.start():
                        active_sessions[call_id] = session
                        log(f"[{call_id}] Session started - listening for speech...")

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

            except json.JSONDecodeError:
                log("Invalid JSON received")
            except Exception as e:
                log(f"Error handling message: {e}")

    except websockets.exceptions.ConnectionClosed:
        log(f"WebSocket closed for call {call_id}")
    except Exception as e:
        log(f"WebSocket error for call {call_id}: {e}")
    finally:
        log(f"[{call_id}] Cleaning up session...")
        if session:
            log(f"[{call_id}] Stopping session and saving audio...")
            await session.stop()
            if call_id in active_sessions:
                del active_sessions[call_id]
            log(f"[{call_id}] Session cleanup complete")


async def main():
    port = int(os.getenv("GEMINI_LIVE_PORT", 8003))

    log("=" * 60)
    log("STT-ONLY SERVICE (No Gemini AI)")
    log("Just transcribes what user says")
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
