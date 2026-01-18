"""
Gemini Live Service for WhatsApp Calls
Uses Gemini Live directly for real-time voice conversations via WebSocket
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

# Overwrite log file on startup
with open(LOG_FILE, "w") as f:
    f.write(f"=== Gemini Live Service Log - Started at {datetime.now()} ===\n\n")

def log(message):
    print(message)
    log_to_file(message)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    log("✅ Loaded environment variables from .env file")
except ImportError:
    log("ℹ️  python-dotenv not installed. Using system environment variables only.")

# Try to add src to path
current_dir = os.path.dirname(__file__)
src_path = os.path.join(current_dir, 'src')
if os.path.exists(src_path):
    sys.path.insert(0, src_path)

# Pipecat imports
try:
    from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
    from pipecat.frames.frames import (
        Frame, StartFrame, EndFrame,
        TTSAudioRawFrame,
        TranscriptionFrame, TextFrame,
        LLMMessagesAppendFrame,
    )
    from pipecat.services.gemini_multimodal_live.gemini import InputAudioRawFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineTask, PipelineParams
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

    # Alias for backward compatibility
    GeminiLiveLLMService = GeminiMultimodalLiveLLMService
    AudioRawFrame = InputAudioRawFrame

    PIPECAT_AVAILABLE = True
    log("✅ Pipecat imports successful (gemini_multimodal_live)")
except ImportError as e:
    log(f"❌ Pipecat import error: {e}")
    import traceback
    log(traceback.format_exc())
    PIPECAT_AVAILABLE = False
    sys.exit(1)

# WebSocket imports
try:
    import websockets
    from websockets.server import serve
    WEBSOCKET_AVAILABLE = True
except ImportError:
    log("❌ websockets not installed. Install with: pip install websockets")
    WEBSOCKET_AVAILABLE = False
    sys.exit(1)

# Load conversation script
CONVERSATION_SCRIPT_PATH = "FAWI_Call_BOT.txt"

def load_conversation_script():
    """Load FAWI_Call_BOT.txt and convert to system prompt"""
    try:
        if os.path.exists(CONVERSATION_SCRIPT_PATH):
            with open(CONVERSATION_SCRIPT_PATH, 'r', encoding='utf-8') as f:
                script_content = f.read()

            system_prompt = f"""You are Mousumi, a Senior Counselor at Freedom with AI. You help people guide their career path using AI skills and how they can make more money out of it.

CONVERSATION SCRIPT:
{script_content}

INSTRUCTIONS:
- Follow the conversation flow from the script above
- Be warm, friendly, and professional
- Ask questions naturally and wait for responses
- Use Indian English accent naturally
- Guide the conversation through connecting questions, situation questions, problem-aware questions, solution-aware questions, and consequence questions
- Present the three pillars when appropriate
- Handle objections professionally
- Keep responses conversational and natural"""

            log(f"✅ Loaded conversation script from {CONVERSATION_SCRIPT_PATH}")
            return system_prompt
        else:
            log(f"⚠️  Conversation script not found: {CONVERSATION_SCRIPT_PATH}")
            return "You are Mousumi, a Senior Counselor at Freedom with AI. Help people with AI skills and career guidance."
    except Exception as e:
        log(f"❌ Error loading conversation script: {e}")
        return "You are Mousumi, a Senior Counselor at Freedom with AI."


class WebSocketAudioOutput(FrameProcessor):
    """Sends audio from Gemini Live to the WebSocket client"""

    def __init__(self, websocket, call_id):
        super().__init__()
        self.websocket = websocket
        self.call_id = call_id
        self.is_active = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Log all frame types for debugging
        frame_type = type(frame).__name__
        if frame_type not in ['SystemFrame', 'MetricsFrame']:
            log(f"🔄 Frame received: {frame_type}")

        # Send audio frames to WebSocket
        if isinstance(frame, TTSAudioRawFrame) and self.is_active:
            try:
                await self.websocket.send(json.dumps({
                    "type": "audio",
                    "data": frame.audio.hex() if isinstance(frame.audio, bytes) else frame.audio
                }))
                log(f"📤 Sent {len(frame.audio)} bytes audio to client")
            except Exception as e:
                log(f"❌ Error sending audio: {e}")

        # Log transcriptions
        if isinstance(frame, TranscriptionFrame):
            text = getattr(frame, 'text', '').strip()
            if text:
                log(f"👤 USER: {text}")

        if isinstance(frame, TextFrame):
            text = getattr(frame, 'text', '').strip()
            if text:
                log(f"🤖 BOT: {text}")

        await self.push_frame(frame, direction)


class WebSocketAudioInput(FrameProcessor):
    """Receives audio from WebSocket and sends to pipeline"""

    def __init__(self, call_id):
        super().__init__()
        self.call_id = call_id
        self.audio_queue = asyncio.Queue()

    async def queue_audio(self, audio_data: bytes):
        """Queue audio from WebSocket to be processed"""
        await self.audio_queue.put(audio_data)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


# Global call sessions
active_calls = {}


async def create_gemini_live_session(websocket, call_id, caller_name):
    """Create Gemini Live session for a call"""
    log(f"🚀 Creating Gemini Live session for call {call_id}")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        log("❌ GOOGLE_API_KEY not found in environment variables")
        return None

    try:
        # Create Gemini Live service with system prompt
        system_prompt = load_conversation_script()

        llm = GeminiLiveLLMService(
            api_key=google_api_key,
            voice_id="Kore",
            system_instruction=system_prompt,
            inference_on_context_initialization=True,  # Make Gemini speak first
        )
        log(f"✅ Gemini Live LLM initialized (Voice: Kore)")

        # Create audio output processor
        audio_output = WebSocketAudioOutput(websocket, call_id)
        audio_input = WebSocketAudioInput(call_id)

        # Create simple pipeline: input -> LLM -> output
        pipeline = Pipeline([
            audio_input,
            llm,
            audio_output,
        ])

        # Create task and runner
        params = PipelineParams(allow_interruptions=True)
        task = PipelineTask(pipeline, params=params)
        runner = PipelineRunner()

        # Store session
        active_calls[call_id] = {
            "llm": llm,
            "audio_input": audio_input,
            "audio_output": audio_output,
            "pipeline": pipeline,
            "task": task,
            "runner": runner,
            "websocket": websocket,
            "caller_name": caller_name,
        }

        # Start pipeline in background
        async def run_pipeline():
            try:
                log(f"🚀 Starting pipeline for call {call_id}")
                await runner.run(task)
                log(f"✅ Pipeline completed for call {call_id}")
            except asyncio.CancelledError:
                log(f"Pipeline cancelled for call {call_id}")
            except Exception as e:
                log(f"❌ Pipeline error for call {call_id}: {e}")
                import traceback
                log(traceback.format_exc())

        pipeline_task = asyncio.create_task(run_pipeline())
        active_calls[call_id]["pipeline_task"] = pipeline_task

        # Wait for pipeline to initialize and Gemini to connect
        await asyncio.sleep(2.0)

        # Send an initial message to trigger Gemini to start speaking
        try:
            # Send a user message to trigger Gemini's greeting
            initial_message = [{"role": "user", "content": "Hello, please greet me and introduce yourself."}]
            messages_frame = LLMMessagesAppendFrame(messages=initial_message)
            await task.queue_frame(messages_frame)
            log(f"📤 Sent initial message to trigger Gemini for call {call_id}")
        except Exception as e:
            log(f"⚠️  Could not send initial message: {e}")
            import traceback
            log(traceback.format_exc())

        log(f"✅ Gemini Live session started for call {call_id}")
        return True

    except Exception as e:
        log(f"❌ Error creating Gemini Live session: {e}")
        import traceback
        log(traceback.format_exc())
        return None


async def handle_websocket(websocket):
    """Handle WebSocket connection from Node.js"""
    call_id = None
    caller_name = None

    try:
        path = getattr(websocket, 'path', '/')
        log(f"📡 WebSocket connection from {websocket.remote_address}")

        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "start":
                    call_id = data.get("call_id")
                    caller_name = data.get("caller_name", "Unknown")

                    if call_id:
                        success = await create_gemini_live_session(websocket, call_id, caller_name)
                        if success:
                            await websocket.send(json.dumps({"type": "started", "call_id": call_id}))
                        else:
                            await websocket.send(json.dumps({"type": "error", "message": "Failed to create session"}))
                    else:
                        await websocket.send(json.dumps({"type": "error", "message": "call_id required"}))

                elif msg_type == "audio":
                    # Receive audio from WhatsApp
                    if call_id and call_id in active_calls:
                        audio_hex = data.get("data", "")
                        if audio_hex:
                            audio_data = bytes.fromhex(audio_hex)
                            audio_input = active_calls[call_id]["audio_input"]

                            # Create audio frame and push to pipeline
                            audio_frame = AudioRawFrame(
                                audio=audio_data,
                                sample_rate=16000,
                                num_channels=1
                            )
                            await audio_input.push_frame(audio_frame, FrameDirection.DOWNSTREAM)
                            log(f"📥 Received {len(audio_data)} bytes audio from client")

                elif msg_type == "stop":
                    if call_id and call_id in active_calls:
                        call_info = active_calls[call_id]

                        # Cancel pipeline
                        if "pipeline_task" in call_info:
                            call_info["pipeline_task"].cancel()

                        del active_calls[call_id]
                        log(f"✅ Call {call_id} stopped")
                        await websocket.send(json.dumps({"type": "stopped", "call_id": call_id}))

            except json.JSONDecodeError:
                log(f"❌ Invalid JSON: {message[:100]}")
            except Exception as e:
                log(f"❌ Error handling message: {e}")
                import traceback
                log(traceback.format_exc())

    except websockets.exceptions.ConnectionClosed:
        log(f"📡 WebSocket closed for call {call_id}")
    except Exception as e:
        log(f"❌ WebSocket error: {e}")
    finally:
        # Cleanup
        if call_id and call_id in active_calls:
            call_info = active_calls[call_id]
            if "pipeline_task" in call_info:
                call_info["pipeline_task"].cancel()
            del active_calls[call_id]


async def main():
    """Start WebSocket server"""
    port = int(os.getenv("GEMINI_LIVE_PORT", 8003))
    log(f"🎙️  Gemini Live Service starting on ws://0.0.0.0:{port}")

    async with serve(handle_websocket, "0.0.0.0", port):
        log(f"✅ Gemini Live Service running on ws://0.0.0.0:{port}")
        log(f"   Waiting for connections from Node.js server...")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("🛑 Server stopped")
