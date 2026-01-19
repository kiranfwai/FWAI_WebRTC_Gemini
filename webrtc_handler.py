"""
WebRTC Handler using aiortc for WhatsApp voice calls

Provides full audio access for bidirectional communication with Gemini Live
"""

import asyncio
import uuid
import fractions
import os
import io
import wave
from typing import Optional, Dict, Any, Callable, Awaitable
from loguru import logger
import numpy as np
import aiohttp

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, RTCConfiguration, RTCIceServer
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame

from config import config
from whatsapp_client import whatsapp_client
from audio_processor import AudioProcessor
from gemini_agent import create_agent, stop_agent, GeminiVoiceAgent


# Sarvam STT for transcription debugging
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
        form_data.add_field('language_code', 'hi-IN')  # Hindi-English mix

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                data=form_data,
                headers={"api-subscription-key": api_key},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    transcript = result.get("transcript", "")
                    return transcript if transcript else "[empty]"
                else:
                    error_text = await response.text()
                    return f"[STT error {response.status}: {error_text[:100]}]"
    except asyncio.TimeoutError:
        return "[STT timeout]"
    except Exception as e:
        return f"[STT error: {str(e)[:50]}]"


class AudioOutputTrack(MediaStreamTrack):
    """
    Custom audio track that outputs audio from Gemini to the caller
    """
    kind = "audio"

    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.sample_rate = sample_rate
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._pts = 0
        self._time_base = fractions.Fraction(1, sample_rate)
        self._frame_samples = 960  # 20ms at 48kHz

        # Buffer for incoming audio
        self._buffer = bytes()
        self._buffer_lock = asyncio.Lock()

    async def recv(self):
        """Generate audio frames for the track"""
        # Target frame size: 20ms at 48kHz = 960 samples
        bytes_needed = self._frame_samples * 2  # 16-bit = 2 bytes per sample

        # Try to get audio from buffer
        async with self._buffer_lock:
            if len(self._buffer) >= bytes_needed:
                audio_bytes = self._buffer[:bytes_needed]
                self._buffer = self._buffer[bytes_needed:]
            else:
                # Generate silence if no audio available
                audio_bytes = bytes(bytes_needed)

        # Convert to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

        # Create AudioFrame
        frame = AudioFrame(format='s16', layout='mono', samples=len(audio_array))
        frame.sample_rate = self.sample_rate
        frame.pts = self._pts
        frame.time_base = self._time_base

        # Copy audio data to frame
        frame.planes[0].update(audio_array.tobytes())

        # Update PTS for next frame
        self._pts += len(audio_array)

        # Wait a bit to maintain timing (~20ms between frames)
        await asyncio.sleep(0.02)

        return frame

    async def push_audio(self, audio_data: bytes, input_sample_rate: int = 24000):
        """
        Push audio data from Gemini to be played to caller

        Args:
            audio_data: PCM audio bytes (mono 16-bit from Gemini)
            input_sample_rate: Sample rate of input audio (default 24kHz from Gemini)
        """
        # Track pushes
        if not hasattr(self, '_push_count'):
            self._push_count = 0
        self._push_count += 1

        # Convert to numpy array
        audio_input = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

        # Calculate resampling ratio
        ratio = self.sample_rate / input_sample_rate
        num_output_samples = int(len(audio_input) * ratio)

        # Resample using linear interpolation
        indices = np.linspace(0, len(audio_input) - 1, num_output_samples)
        audio_resampled = np.interp(indices, np.arange(len(audio_input)), audio_input)
        audio_resampled = np.clip(audio_resampled, -32768, 32767).astype(np.int16)

        async with self._buffer_lock:
            self._buffer += audio_resampled.tobytes()
            if self._push_count <= 5 or self._push_count % 50 == 0:
                logger.info(f"[OUTPUT_TRACK] push #{self._push_count}: {len(audio_data)}b in -> {len(audio_resampled)*2}b buffered, total buffer: {len(self._buffer)} bytes")


class AudioTrackReader:
    """
    Reads audio from a WebRTC MediaStreamTrack and sends to Gemini
    """

    def __init__(
        self,
        track: MediaStreamTrack,
        audio_processor: AudioProcessor,
        on_audio: Callable[[bytes], Awaitable[None]]
    ):
        self.track = track
        self.audio_processor = audio_processor
        self.on_audio = on_audio
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Transcription buffer - accumulate audio for STT
        self._transcription_buffer = bytes()
        self._is_speaking = False
        self._silence_start = None
        self._transcription_count = 0

        # DEBUG: Save raw 48kHz audio for analysis
        self._raw_48k_buffer = bytes()
        self._call_id_short = None

    async def start(self):
        """Start reading audio from the track"""
        self._running = True
        self._task = asyncio.create_task(self._read_loop())
        logger.info("Audio track reader started")

    async def stop(self):
        """Stop reading audio"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # DEBUG: Save raw 48kHz audio for analysis
        if len(self._raw_48k_buffer) > 0:
            try:
                call_id = self._call_id_short or "unknown"
                filename = f"/app/audio_debug/raw_48k_{call_id}.wav"
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(48000)
                    wf.writeframes(self._raw_48k_buffer)
                with open(filename, 'wb') as f:
                    f.write(wav_buffer.getvalue())
                duration = len(self._raw_48k_buffer) / (48000 * 2)
                logger.info(f"📁 RAW 48kHz AUDIO SAVED: {filename} ({duration:.1f}s)")
            except Exception as e:
                logger.error(f"Error saving raw 48k audio: {e}")

        logger.info("Audio track reader stopped")

    async def _transcribe_and_log(self, audio_data: bytes):
        """Transcribe accumulated audio and log what user said"""
        try:
            self._transcription_count += 1
            # Audio is at 16kHz after resampling
            transcript = await transcribe_audio_sarvam(audio_data, sample_rate=16000)
            logger.info(f"╔══════════════════════════════════════════════════════════════╗")
            logger.info(f"║ 🎤 USER SPEECH #{self._transcription_count}: \"{transcript}\"")
            logger.info(f"║ Audio: {len(audio_data)} bytes ({len(audio_data)/32:.1f}ms at 16kHz)")
            logger.info(f"╚══════════════════════════════════════════════════════════════╝")
        except Exception as e:
            logger.error(f"Transcription error: {e}")

    async def _read_loop(self):
        """Main loop to read audio frames"""
        frame_count = 0
        try:
            while self._running:
                try:
                    # Receive frame from track
                    frame = await self.track.recv()
                    frame_count += 1

                    # Extract audio data from frame using planes (raw bytes)
                    # This is more reliable than to_ndarray() for some codecs
                    if hasattr(frame, 'planes') and len(frame.planes) > 0:
                        # Get raw bytes directly from the audio plane
                        raw_audio_bytes = bytes(frame.planes[0])

                        # Determine format and convert to int16 array
                        sample_rate = getattr(frame, 'sample_rate', 48000)
                        fmt = getattr(frame, 'format', None)
                        fmt_name = str(fmt.name) if fmt else 'unknown'

                        # Get channel info
                        channels = getattr(frame.layout, 'channels', None)
                        num_channels = len(channels) if channels else 1

                        # Log frame info for debugging (first few frames)
                        if frame_count <= 10:
                            logger.info(f"[WEBRTC_FRAME] #{frame_count}: planes={len(frame.planes)}, "
                                       f"plane_size={len(raw_audio_bytes)}, format={fmt_name}, "
                                       f"sample_rate={sample_rate}, samples={frame.samples}, channels={num_channels}")

                        # Convert based on format
                        if 's16' in fmt_name:
                            # Signed 16-bit PCM
                            audio_array = np.frombuffer(raw_audio_bytes, dtype=np.int16)
                        elif 's32' in fmt_name:
                            # Signed 32-bit PCM - convert to 16-bit
                            audio_array = np.frombuffer(raw_audio_bytes, dtype=np.int32)
                            audio_array = (audio_array / 65536).astype(np.int16)
                        elif 'flt' in fmt_name or 'f32' in fmt_name:
                            # 32-bit float [-1.0, 1.0]
                            audio_array = np.frombuffer(raw_audio_bytes, dtype=np.float32)
                            audio_array = (audio_array * 32767).astype(np.int16)
                        elif 'dbl' in fmt_name or 'f64' in fmt_name:
                            # 64-bit float
                            audio_array = np.frombuffer(raw_audio_bytes, dtype=np.float64)
                            audio_array = (audio_array * 32767).astype(np.int16)
                        else:
                            # Default: assume 16-bit
                            audio_array = np.frombuffer(raw_audio_bytes, dtype=np.int16)

                        # Handle stereo -> mono conversion
                        if num_channels == 2 and len(audio_array) > 0:
                            # Deinterleave stereo (L R L R...) and average to mono
                            left = audio_array[0::2]
                            right = audio_array[1::2]
                            audio_array = ((left.astype(np.int32) + right.astype(np.int32)) // 2).astype(np.int16)
                            if frame_count <= 10:
                                logger.info(f"[WEBRTC_STEREO] Converted stereo to mono: {len(left)*2} -> {len(audio_array)} samples")

                        # Log audio level
                        if frame_count <= 10:
                            raw_max = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0
                            logger.info(f"[WEBRTC_AUDIO] #{frame_count}: samples={len(audio_array)}, raw_max={raw_max}")

                        # DEBUG: Save raw audio
                        self._raw_48k_buffer += audio_array.tobytes()

                        # Convert to float for processing
                        audio_float = audio_array.astype(np.float32)

                        # AMPLIFY: WhatsApp/WebRTC audio is very quiet after stereo->mono (raw levels 1-100)
                        # Apply strong fixed gain since adaptive gain doesn't work well with such low levels
                        max_val_before = np.max(np.abs(audio_float)) if len(audio_float) > 0 else 0

                        # Use fixed high gain - WhatsApp stereo audio is extremely quiet
                        # Target ~20000 amplitude for speech, gain of 500-1000x needed for raw levels of 20-40
                        FIXED_GAIN = 800.0
                        audio_float = audio_float * FIXED_GAIN

                        # Soft clip to avoid harsh distortion
                        audio_int16 = np.clip(audio_float, -32768, 32767).astype(np.int16)

                        # Log audio level for first frames
                        if frame_count <= 10:
                            max_val_after = np.max(np.abs(audio_int16))
                            logger.info(f"[WEBRTC_AUDIO] #{frame_count}: samples={len(audio_int16)}, before={max_val_before:.0f}, after={max_val_after}")

                        # Convert to bytes
                        audio_bytes = audio_int16.tobytes()
                        
                        # ===== REAL-TIME SPEECH DETECTION LOGGING =====
                        max_val_after = np.max(np.abs(audio_int16)) if len(audio_int16) > 0 else 0
                        if frame_count % 50 == 0 or max_val_after > 500:
                            is_speech = "🗣️ SPEECH DETECTED!" if max_val_after > 500 else "silence"
                            print(f"[FRAME {frame_count}] Audio level: {max_val_after:.0f} - {is_speech}")

                        # Process and send to callback (resamples to 16kHz)
                        processed = self.audio_processor.process_webrtc_audio(audio_bytes)
                        if processed and self.on_audio:
                            await self.on_audio(processed)

                        # Note: Transcription is handled in gemini-live-service.py to avoid duplicates

                except Exception as e:
                    if self._running:
                        logger.error(f"Error reading audio frame: {e}")
                        import traceback
                        if frame_count <= 5:
                            logger.error(traceback.format_exc())
                    await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in audio read loop: {e}")


class CallSession:
    """
    Manages a single WhatsApp voice call session

    Handles WebRTC connection, audio processing, and Gemini integration
    """

    def __init__(
        self,
        call_id: str,
        phone_number: str,
        caller_name: str = "Customer",
        is_outbound: bool = False
    ):
        self.call_id = call_id
        self.phone_number = phone_number
        self.caller_name = caller_name
        self.is_outbound = is_outbound

        # WebRTC
        self.pc: Optional[RTCPeerConnection] = None

        # Audio processing
        self.audio_processor: Optional[AudioProcessor] = None
        self.track_reader: Optional[AudioTrackReader] = None

        # Audio output track (sends audio to caller)
        self.output_track: Optional[AudioOutputTrack] = None

        # Gemini agent
        self.agent: Optional[GeminiVoiceAgent] = None

        # Audio output buffer
        self._audio_output_queue: asyncio.Queue = asyncio.Queue()

        # State
        self._running = False

    async def setup_peer_connection(self):
        """Create and configure the RTCPeerConnection"""
        ice_servers = [RTCIceServer(urls=s["urls"]) for s in config.ice_servers]
        rtc_config = RTCConfiguration(iceServers=ice_servers)

        self.pc = RTCPeerConnection(configuration=rtc_config)

        # Handle incoming audio track (from WhatsApp caller)
        @self.pc.on("track")
        async def on_track(track):
            logger.info(f"Received track: {track.kind}")
            if track.kind == "audio":
                await self._handle_audio_track(track)

        # Handle ICE candidates
        @self.pc.on("icecandidate")
        async def on_ice_candidate(candidate):
            if candidate:
                logger.debug(f"ICE candidate: {candidate.candidate}")
                await whatsapp_client.send_ice_candidate(
                    self.call_id,
                    {
                        "candidate": candidate.candidate,
                        "sdpMid": candidate.sdpMid,
                        "sdpMLineIndex": candidate.sdpMLineIndex
                    }
                )

        # Handle connection state changes
        @self.pc.on("connectionstatechange")
        async def on_connection_state_change():
            logger.info(f"Connection state: {self.pc.connectionState}")
            if self.pc.connectionState == "connected":
                logger.info(f"Call {self.call_id} connected!")
            elif self.pc.connectionState in ["failed", "closed", "disconnected"]:
                logger.info(f"Call {self.call_id} ended: {self.pc.connectionState}")
                await self.stop()

        logger.info(f"Peer connection setup for call {self.call_id}")

    async def _handle_audio_track(self, track: MediaStreamTrack):
        """Handle incoming audio track from WhatsApp"""
        logger.info(f"Setting up audio processing for call {self.call_id}")

        # Create audio processor
        self.audio_processor = AudioProcessor()
        await self.audio_processor.start()

        # Track audio from WhatsApp
        self._whatsapp_audio_count = 0

        # Create track reader - sends audio to Gemini Live via WebSocket
        async def on_audio(pcm_data: bytes):
            self._whatsapp_audio_count += 1

            # Log audio levels (not just non-zero count)
            if self._whatsapp_audio_count <= 10 or self._whatsapp_audio_count % 50 == 0:
                audio_arr = np.frombuffer(pcm_data, dtype=np.int16)
                max_level = np.max(np.abs(audio_arr)) if len(audio_arr) > 0 else 0
                logger.info(f"[TO_GEMINI][{self.call_id}] #{self._whatsapp_audio_count}: {len(pcm_data)} bytes, max_level={max_level}")

            if self.agent:
                await self.agent.send_audio(pcm_data)

        self.track_reader = AudioTrackReader(track, self.audio_processor, on_audio)
        self.track_reader._call_id_short = self.call_id[:8] if len(self.call_id) > 8 else self.call_id
        await self.track_reader.start()

        logger.info(f"Audio processing started for call {self.call_id}")

    async def _handle_gemini_audio(self, audio_data: bytes, sample_rate: int = 24000):
        """Handle audio output from Gemini - sends to caller via WebRTC"""
        # Track audio output
        if not hasattr(self, '_gemini_out_count'):
            self._gemini_out_count = 0
        self._gemini_out_count += 1

        if self.output_track:
            await self.output_track.push_audio(audio_data, sample_rate)
            if self._gemini_out_count <= 5 or self._gemini_out_count % 50 == 0:
                logger.info(f"[GEMINI_TO_WEBRTC] #{self._gemini_out_count}: {len(audio_data)} bytes @ {sample_rate}Hz -> output_track")
        else:
            if self._gemini_out_count <= 5 or self._gemini_out_count % 50 == 0:
                logger.warning(f"[GEMINI_TO_WEBRTC] #{self._gemini_out_count}: No output_track! Audio dropped: {len(audio_data)} bytes")

    async def start_agent(self):
        """Start the Gemini voice agent"""
        self.agent = await create_agent(
            call_id=self.call_id,
            caller_name=self.caller_name,
            on_audio_output=self._handle_gemini_audio
        )

        if self.agent:
            logger.info(f"Gemini agent started for call {self.call_id}")
            return True
        else:
            logger.error(f"Failed to start Gemini agent for call {self.call_id}")
            return False

    async def create_offer(self) -> str:
        """Create SDP offer for outbound call"""
        await self.setup_peer_connection()

        # Create output track for sending audio to caller
        self.output_track = AudioOutputTrack(sample_rate=48000)

        # Add the audio track to the peer connection
        self.pc.addTrack(self.output_track)

        # Create offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # Wait for ICE gathering
        await self._wait_for_ice_gathering()

        return self.pc.localDescription.sdp

    async def handle_answer(self, sdp_answer: str):
        """Handle SDP answer from WhatsApp"""
        answer = RTCSessionDescription(sdp=sdp_answer, type="answer")
        await self.pc.setRemoteDescription(answer)
        logger.info(f"Remote description set for call {self.call_id}")

    async def handle_offer(self, sdp_offer: str) -> str:
        """Handle incoming call with SDP offer"""
        await self.setup_peer_connection()

        # Create output track for sending audio to caller
        self.output_track = AudioOutputTrack(sample_rate=48000)

        # Add the audio track to the peer connection
        self.pc.addTrack(self.output_track)

        # Set remote description (offer from WhatsApp)
        offer = RTCSessionDescription(sdp=sdp_offer, type="offer")
        await self.pc.setRemoteDescription(offer)

        # Create answer
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)

        # Wait for ICE gathering
        await self._wait_for_ice_gathering()

        return self.pc.localDescription.sdp

    async def add_ice_candidate(self, candidate_data: Dict[str, Any]):
        """Add ICE candidate from WhatsApp"""
        try:
            candidate = RTCIceCandidate(
                candidate=candidate_data.get("candidate"),
                sdpMid=candidate_data.get("sdpMid"),
                sdpMLineIndex=candidate_data.get("sdpMLineIndex")
            )
            await self.pc.addIceCandidate(candidate)
        except Exception as e:
            logger.error(f"Error adding ICE candidate: {e}")

    async def _wait_for_ice_gathering(self, timeout: float = 5.0):
        """Wait for ICE gathering to complete"""
        if self.pc.iceGatheringState == "complete":
            return

        gathering_complete = asyncio.Event()

        @self.pc.on("icegatheringstatechange")
        def check_state():
            if self.pc.iceGatheringState == "complete":
                gathering_complete.set()

        try:
            await asyncio.wait_for(gathering_complete.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("ICE gathering timed out")

    async def stop(self):
        """Stop the call session"""
        logger.info(f"Stopping call session {self.call_id}")
        self._running = False

        # Stop track reader
        if self.track_reader:
            await self.track_reader.stop()

        # Stop output track
        if self.output_track:
            self.output_track.stop()

        # Stop audio processor
        if self.audio_processor:
            await self.audio_processor.stop()

        # Stop Gemini agent
        if self.agent:
            await self.agent.stop()  # Call stop directly on the agent object

        # Close peer connection
        if self.pc:
            await self.pc.close()

        logger.info(f"Call session {self.call_id} stopped")


# Store active call sessions
active_sessions: Dict[str, CallSession] = {}


async def make_outbound_call(
    phone_number: str,
    caller_name: str = "Customer"
) -> Dict[str, Any]:
    """
    Make an outbound call to a WhatsApp user
    """
    call_id = str(uuid.uuid4())

    logger.info(f"Initiating outbound call to {phone_number}")

    session = CallSession(
        call_id=call_id,
        phone_number=phone_number,
        caller_name=caller_name,
        is_outbound=True
    )

    try:
        # Create SDP offer
        sdp_offer = await session.create_offer()

        # Start Gemini agent
        await session.start_agent()

        # Make call via WhatsApp API
        result = await whatsapp_client.make_call(
            phone_number=phone_number,
            sdp_offer=sdp_offer
        )

        if result.get("success"):
            wa_call_id = result.get("calls", [{}])[0].get("id", call_id)
            session.call_id = wa_call_id
            active_sessions[wa_call_id] = session

            sdp_answer = result.get("calls", [{}])[0].get("sdp")
            if sdp_answer:
                await session.handle_answer(sdp_answer)

            logger.info(f"Outbound call initiated: {wa_call_id}")
            return {
                "success": True,
                "call_id": wa_call_id,
                "message": "Call initiated"
            }
        else:
            await session.stop()
            return {
                "success": False,
                "error": result.get("error", "Unknown error")
            }

    except Exception as e:
        logger.error(f"Error making outbound call: {e}")
        import traceback
        traceback.print_exc()
        await session.stop()
        return {
            "success": False,
            "error": str(e)
        }


async def handle_incoming_call(
    call_id: str,
    caller_phone: str,
    sdp_offer: str,
    caller_name: str = "Customer"
) -> Dict[str, Any]:
    """Handle an incoming call from WhatsApp"""
    logger.info(f"Handling incoming call {call_id} from {caller_phone}")

    session = CallSession(
        call_id=call_id,
        phone_number=caller_phone,
        caller_name=caller_name,
        is_outbound=False
    )

    try:
        sdp_answer = await session.handle_offer(sdp_offer)
        await session.start_agent()

        result = await whatsapp_client.answer_call(
            call_id=call_id,
            sdp_answer=sdp_answer
        )

        if result.get("success"):
            active_sessions[call_id] = session
            logger.info(f"Incoming call answered: {call_id}")
            return {
                "success": True,
                "call_id": call_id,
                "message": "Call answered"
            }
        else:
            await session.stop()
            return {
                "success": False,
                "error": result.get("error", "Failed to answer call")
            }

    except Exception as e:
        logger.error(f"Error handling incoming call: {e}")
        await session.stop()
        return {
            "success": False,
            "error": str(e)
        }


async def handle_ice_candidate(call_id: str, candidate: Dict[str, Any]):
    """Handle ICE candidate from WhatsApp"""
    session = active_sessions.get(call_id)
    if session:
        await session.add_ice_candidate(candidate)


async def terminate_call(call_id: str) -> Dict[str, Any]:
    """Terminate a call"""
    session = active_sessions.get(call_id)
    if session:
        await session.stop()
        del active_sessions[call_id]
        await whatsapp_client.terminate_call(call_id)
        return {"success": True, "message": "Call terminated"}

    return {"success": False, "error": "Call not found"}


def get_active_calls() -> list:
    """Get list of active calls"""
    return [
        {
            "call_id": call_id,
            "phone_number": session.phone_number,
            "is_outbound": session.is_outbound
        }
        for call_id, session in active_sessions.items()
    ]
