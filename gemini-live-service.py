"""
Gemini Live Service for WhatsApp Calls
Uses the SAME logic as test_gemini_live_working.py - just with WebSocket I/O instead of Daily transport
"""

import os
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

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

# Pipecat imports - SAME as test_gemini_live_working.py
try:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineTask, PipelineParams
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
    from pipecat.frames.frames import (
        Frame,
        InputAudioRawFrame,
        OutputAudioRawFrame,
        LLMMessagesFrame,
        LLMMessagesUpdateFrame,
        StartFrame, EndFrame,
        TranscriptionFrame, LLMTextFrame, TTSAudioRawFrame
    )
    from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
    from pipecat.transports.base_input import BaseInputTransport
    from pipecat.transports.base_output import BaseOutputTransport

    log("Pipecat imports successful (same as test_gemini_live_working.py)")
except ImportError as e:
    log(f"Pipecat import error: {e}")
    import traceback
    log(traceback.format_exc())
    sys.exit(1)

# WebSocket imports
try:
    import websockets
    from websockets.server import serve
except ImportError:
    log("websockets not installed")
    sys.exit(1)

# Load conversation script
CONVERSATION_SCRIPT_PATH = "FAWI_Call_BOT.txt"

def load_conversation_script():
    """Load FAWI_Call_BOT.txt - optimized for Freedom with AI live agent"""
    try:
        # Optimized system prompt for faster, smarter responses
        system_prompt = """You are Mousumi, a Senior AI Counselor at Freedom with AI.

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

        log("Loaded optimized conversation script for Freedom with AI agent")
        return system_prompt
    except Exception as e:
        log(f"Error loading conversation script: {e}")
        return "You are Mousumi, a Senior AI Counselor at Freedom with AI. Use Indian English accent."


class WebSocketInputTransport(FrameProcessor):
    """
    Replaces transport.input() from Daily transport.
    Receives audio from WebSocket and produces InputAudioRawFrame into pipeline.
    """
    def __init__(self, call_id):
        super().__init__()
        self.call_id = call_id
        self._audio_queue = asyncio.Queue()
        self._running = False
        self._frame_count = 0

    async def start(self):
        """Start producing frames into the pipeline."""
        self._running = True
        log(f"[{self.call_id}] WebSocket input transport started")

    async def stop(self):
        """Stop the transport."""
        self._running = False
        await self._audio_queue.put(None)  # Signal to stop

    async def push_audio(self, audio_data: bytes):
        """Push audio from WebSocket into the queue."""
        if self._running:
            await self._audio_queue.put(audio_data)

    async def run_input_loop(self, task: PipelineTask):
        """
        Background task that reads from audio queue and pushes frames into pipeline.
        This mimics what transport.input() does in Daily transport.
        """
        log(f"[{self.call_id}] Starting input loop")
        while self._running:
            try:
                audio_data = await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
                if audio_data is None:
                    break

                # Create and queue InputAudioRawFrame - same as Daily transport does
                frame = InputAudioRawFrame(
                    audio=audio_data,
                    sample_rate=16000,
                    num_channels=1
                )
                self._frame_count += 1
                if self._frame_count <= 3 or self._frame_count % 100 == 0:
                    log(f"[{self.call_id}] Queued input frame #{self._frame_count}: {len(audio_data)} bytes")
                await task.queue_frame(frame)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log(f"[{self.call_id}] Input loop error: {e}")
                break
        log(f"[{self.call_id}] Input loop ended")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Pass through frames."""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class WebSocketOutputTransport(FrameProcessor):
    """
    Replaces transport.output() from Daily transport.
    Receives TTSAudioRawFrame/OutputAudioRawFrame and sends to WebSocket.
    """
    def __init__(self, websocket, call_id):
        super().__init__()
        self.websocket = websocket
        self.call_id = call_id
        self._running = True
        self._output_count = 0

    async def stop(self):
        self._running = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and send audio to WebSocket."""
        await super().process_frame(frame, direction)

        # Send audio frames to WebSocket - same types that Daily transport.output() handles
        if isinstance(frame, (TTSAudioRawFrame, OutputAudioRawFrame)) and self._running:
            try:
                audio_data = frame.audio
                if isinstance(audio_data, bytes):
                    audio_hex = audio_data.hex()
                else:
                    audio_hex = bytes(audio_data).hex()

                sample_rate = getattr(frame, 'sample_rate', 24000)

                self._output_count += 1

                await self.websocket.send(json.dumps({
                    "type": "audio",
                    "data": audio_hex,
                    "sample_rate": sample_rate,
                    "num_channels": 1
                }))
                if self._output_count <= 5 or self._output_count % 50 == 0:
                    log(f"[{self.call_id}] Sent output #{self._output_count}: {len(audio_data)} bytes at {sample_rate}Hz")
            except Exception as e:
                log(f"[{self.call_id}] Error sending audio: {e}")

        await self.push_frame(frame, direction)


class ChatLogger(FrameProcessor):
    """
    SAME as test_gemini_live_working.py - logs conversation flow.
    """
    def __init__(self, call_id):
        super().__init__()
        self.call_id = call_id
        self._bot_buffer = ""
        self._frame_count = 0
        self._audio_frame_count = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        self._frame_count += 1

        # Log frame types for debugging (first few and then periodically)
        if self._frame_count <= 10 or self._frame_count % 200 == 0:
            log(f"[{self.call_id}] ChatLogger frame #{self._frame_count}: {type(frame).__name__} (dir={direction})")

        # Log User Input
        if isinstance(frame, TranscriptionFrame):
            text = getattr(frame, 'text', '').strip()
            if text:
                log(f"[{self.call_id}] USER: {text}")

        # Log Bot Output
        elif isinstance(frame, LLMTextFrame):
            self._bot_buffer += frame.text
            log(f"[{self.call_id}] LLMTextFrame: {frame.text[:100]}...")

        elif isinstance(frame, LLMMessagesFrame):
            if self._bot_buffer.strip():
                log(f"[{self.call_id}] BOT: {self._bot_buffer.strip()}")
                self._bot_buffer = ""

        # Log audio output frames
        elif isinstance(frame, (TTSAudioRawFrame, OutputAudioRawFrame)):
            self._audio_frame_count += 1
            if self._audio_frame_count <= 3 or self._audio_frame_count % 50 == 0:
                log(f"[{self.call_id}] Audio output frame #{self._audio_frame_count}: {len(frame.audio)} bytes")

        await self.push_frame(frame)


# Global sessions
active_calls = {}


async def create_gemini_session(websocket, call_id: str, caller_name: str):
    """
    Create Gemini Live session using SAME structure as test_gemini_live_working.py
    """
    log(f"[{call_id}] Creating Gemini Live session (same logic as working implementation)")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        log(f"[{call_id}] GOOGLE_API_KEY not found")
        return None

    try:
        # --- SAME as test_gemini_live_working.py ---

        # 1. Load system prompt first
        system_prompt = load_conversation_script()

        # 2. Create LLM with system_instruction (pass directly to Gemini Live)
        llm = GeminiLiveLLMService(
            api_key=google_api_key,
            voice_id="Charon",  # Same voice as working implementation
            system_instruction=system_prompt  # Pass system instruction directly
        )
        log(f"[{call_id}] Gemini Service Ready (Voice: Charon) with system_instruction")

        # 3. Create context and aggregator (for conversation history)
        messages = [{"role": "system", "content": system_prompt}]
        context = LLMContext(messages=messages)
        context_aggregator = LLMContextAggregatorPair(context)
        log(f"[{call_id}] Created LLMContext and context aggregator")

        # 3. Create transports (WebSocket versions of transport.input()/output())
        input_transport = WebSocketInputTransport(call_id)
        output_transport = WebSocketOutputTransport(websocket, call_id)
        chat_logger = ChatLogger(call_id)

        # 4. Create pipeline - EXACT SAME STRUCTURE as test_gemini_live_working.py:
        #    transport.input() -> context_aggregator.user() -> llm -> ChatLogger -> transport.output() -> context_aggregator.assistant()
        pipeline = Pipeline([
            input_transport,              # = transport.input()
            context_aggregator.user(),    # SAME
            llm,                          # SAME
            chat_logger,                  # SAME as ChatLogger()
            output_transport,             # = transport.output()
            context_aggregator.assistant() # SAME
        ])
        log(f"[{call_id}] Pipeline created (same structure as working implementation)")

        # 5. Create task and runner (with heartbeats enabled to prevent idle timeout)
        task = PipelineTask(pipeline, params=PipelineParams(
            allow_interruptions=True,
            enable_heartbeats=True,
            heartbeats_period_secs=5.0
        ))
        runner = PipelineRunner(handle_sigint=False)

        # Store session
        session = {
            "llm": llm,
            "context": context,
            "context_aggregator": context_aggregator,
            "input_transport": input_transport,
            "output_transport": output_transport,
            "pipeline": pipeline,
            "task": task,
            "runner": runner,
            "websocket": websocket,
            "caller_name": caller_name,
        }
        active_calls[call_id] = session

        # Start input transport
        await input_transport.start()

        # Start input loop in background (reads from queue and pushes to pipeline)
        input_loop_task = asyncio.create_task(input_transport.run_input_loop(task))
        session["input_loop_task"] = input_loop_task

        # Start pipeline in background
        async def run_pipeline():
            try:
                log(f"[{call_id}] Starting pipeline runner")
                await runner.run(task)
                log(f"[{call_id}] Pipeline completed")
            except asyncio.CancelledError:
                log(f"[{call_id}] Pipeline cancelled")
            except Exception as e:
                log(f"[{call_id}] Pipeline error: {e}")
                import traceback
                log(traceback.format_exc())

        pipeline_task = asyncio.create_task(run_pipeline())
        session["pipeline_task"] = pipeline_task

        # Wait briefly for pipeline to initialize (reduced for faster response)
        await asyncio.sleep(0.3)

        # 6. Trigger greeting - SAME as test_gemini_live_working.py trigger_instant_greeting()
        user_ref = f"Hi {caller_name}" if caller_name and caller_name != "Unknown" else "Hi"
        greeting_text = f"{user_ref}, this is Mousumi from Freedom with AI. I noticed you showed interest in our AI programs. How are you doing today?"

        log(f"[{call_id}] Triggering greeting (same as working implementation)")

        # Force Generation with System Command - using new pipecat API
        trigger_msg = {
            "role": "user",
            "content": f"SYSTEM_COMMAND: The user has joined. Speak this EXACT greeting immediately with Indian Accent: '{greeting_text}'"
        }

        # Use LLMMessagesUpdateFrame with run_llm=True for pipecat 0.0.99+
        await task.queue_frames([
            LLMMessagesUpdateFrame(messages=[trigger_msg], run_llm=True)
        ])

        log(f"[{call_id}] Greeting signal sent!")
        return session

    except Exception as e:
        log(f"[{call_id}] Error creating session: {e}")
        import traceback
        log(traceback.format_exc())
        return None


async def handle_websocket(websocket):
    """Handle WebSocket connection from main server."""
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
                        session = await create_gemini_session(websocket, call_id, caller_name)
                        if session:
                            await websocket.send(json.dumps({"type": "started", "call_id": call_id}))
                        else:
                            await websocket.send(json.dumps({"type": "error", "message": "Failed to create session"}))
                    else:
                        await websocket.send(json.dumps({"type": "error", "message": "call_id required"}))

                elif msg_type == "audio":
                    # Receive audio from WhatsApp and push to input transport
                    if session and call_id in active_calls:
                        audio_hex = data.get("data", "")
                        if audio_hex:
                            audio_data = bytes.fromhex(audio_hex)
                            # Log first few audio messages
                            if not hasattr(session, '_audio_count'):
                                session['_audio_count'] = 0
                            session['_audio_count'] += 1
                            if session['_audio_count'] <= 5 or session['_audio_count'] % 100 == 0:
                                log(f"[{call_id}] Received audio #{session['_audio_count']}: {len(audio_data)} bytes")
                            # Push to input transport - it will create InputAudioRawFrame
                            await session["input_transport"].push_audio(audio_data)
                    else:
                        log(f"[{call_id}] Audio received but no session or not in active_calls")

                elif msg_type == "stop":
                    if session and call_id in active_calls:
                        # Stop input transport
                        await session["input_transport"].stop()
                        await session["output_transport"].stop()

                        # Cancel tasks
                        if "input_loop_task" in session:
                            session["input_loop_task"].cancel()
                        if "pipeline_task" in session:
                            session["pipeline_task"].cancel()

                        del active_calls[call_id]
                        log(f"[{call_id}] Call stopped")
                        await websocket.send(json.dumps({"type": "stopped", "call_id": call_id}))

            except json.JSONDecodeError:
                log(f"Invalid JSON: {message[:100]}")
            except Exception as e:
                log(f"Error handling message: {e}")
                import traceback
                log(traceback.format_exc())

    except websockets.exceptions.ConnectionClosed:
        log(f"WebSocket closed for call {call_id}")
    except Exception as e:
        log(f"WebSocket error: {e}")
    finally:
        if session and call_id in active_calls:
            await session["input_transport"].stop()
            await session["output_transport"].stop()
            if "input_loop_task" in session:
                session["input_loop_task"].cancel()
            if "pipeline_task" in session:
                session["pipeline_task"].cancel()
            del active_calls[call_id]


async def main():
    """Start WebSocket server"""
    port = int(os.getenv("GEMINI_LIVE_PORT", 8003))
    log(f"Gemini Live Service starting on ws://0.0.0.0:{port}")
    log(f"Using SAME pipeline logic as test_gemini_live_working.py")

    async with serve(handle_websocket, "0.0.0.0", port):
        log(f"Gemini Live Service running on ws://0.0.0.0:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Server stopped")
