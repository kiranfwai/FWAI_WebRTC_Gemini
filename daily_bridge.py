"""
Daily Bridge - Bridges WhatsApp WebRTC audio to Daily room
Uses daily-python SDK to join as a participant and stream audio
"""

import asyncio
import numpy as np
from typing import Optional, Callable, Awaitable
from loguru import logger

try:
    from daily import Daily, CallClient, EventHandler
    DAILY_AVAILABLE = True
except ImportError:
    DAILY_AVAILABLE = False
    logger.warning("daily-python not installed. Install with: pip install daily-python")


class DailyBridge(EventHandler):
    """
    Bridges audio between WhatsApp WebRTC and Daily room

    - Joins Daily room as a participant
    - Sends WhatsApp audio to Daily room
    - Receives Daily room audio and sends back to WhatsApp
    """

    def __init__(
        self,
        room_url: str,
        call_id: str,
        on_audio_from_daily: Optional[Callable[[bytes, int], Awaitable[None]]] = None
    ):
        super().__init__()
        self.room_url = room_url
        self.call_id = call_id
        self.on_audio_from_daily = on_audio_from_daily

        self.client: Optional[CallClient] = None
        self._running = False
        self._joined = asyncio.Event()

        # Audio settings
        self.sample_rate = 16000  # Daily uses 16kHz
        self.channels = 1

    async def join(self) -> bool:
        """Join the Daily room"""
        if not DAILY_AVAILABLE:
            logger.error("daily-python not available")
            return False

        try:
            # Initialize Daily
            Daily.init()

            # Create call client
            self.client = CallClient(event_handler=self)

            # Join room
            logger.info(f"[{self.call_id}] Joining Daily room: {self.room_url}")
            self.client.join(
                self.room_url,
                client_settings={
                    "inputs": {
                        "camera": False,
                        "microphone": {
                            "isEnabled": True,
                            "settings": {
                                "deviceId": "virtual",  # Virtual mic - we'll send audio programmatically
                            }
                        }
                    },
                    "publishing": {
                        "camera": {"isPublishing": False},
                        "microphone": {"isPublishing": True}
                    }
                },
                completion=self._on_join_complete
            )

            # Wait for join to complete
            try:
                await asyncio.wait_for(self._joined.wait(), timeout=10.0)
                self._running = True
                logger.info(f"[{self.call_id}] Joined Daily room successfully")
                return True
            except asyncio.TimeoutError:
                logger.error(f"[{self.call_id}] Timeout joining Daily room")
                return False

        except Exception as e:
            logger.error(f"[{self.call_id}] Error joining Daily room: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _on_join_complete(self, data, error):
        """Callback when join completes"""
        if error:
            logger.error(f"[{self.call_id}] Join error: {error}")
        else:
            logger.info(f"[{self.call_id}] Join complete: {data}")
            # Set event in thread-safe way
            asyncio.get_event_loop().call_soon_threadsafe(self._joined.set)

    def on_participant_joined(self, participant):
        """Called when a participant joins"""
        logger.info(f"[{self.call_id}] Participant joined: {participant.get('id', 'unknown')}")

    def on_participant_left(self, participant, reason):
        """Called when a participant leaves"""
        logger.info(f"[{self.call_id}] Participant left: {participant.get('id', 'unknown')}")

    def on_audio_data(self, participant_id, audio_data):
        """
        Called when audio is received from the Daily room
        This is audio from the Gemini bot
        """
        if not self._running:
            return

        if self.on_audio_from_daily:
            # audio_data is typically int16 PCM at 16kHz
            asyncio.create_task(
                self.on_audio_from_daily(audio_data, self.sample_rate)
            )

    async def send_audio(self, audio_data: bytes):
        """
        Send audio to the Daily room
        This is audio from WhatsApp caller

        Args:
            audio_data: PCM audio bytes (16kHz, mono, 16-bit)
        """
        if not self._running or not self.client:
            return

        try:
            # Send audio to Daily room
            # daily-python expects audio frames in a specific format
            self.client.send_app_message({
                "type": "audio",
                "data": audio_data.hex()
            })
        except Exception as e:
            logger.error(f"[{self.call_id}] Error sending audio to Daily: {e}")

    async def leave(self):
        """Leave the Daily room"""
        logger.info(f"[{self.call_id}] Leaving Daily room")
        self._running = False

        if self.client:
            try:
                self.client.leave()
            except:
                pass

        logger.info(f"[{self.call_id}] Left Daily room")


async def create_daily_bridge(
    room_url: str,
    call_id: str,
    on_audio_from_daily: Optional[Callable[[bytes, int], Awaitable[None]]] = None
) -> Optional[DailyBridge]:
    """
    Create and connect a Daily bridge

    Args:
        room_url: Daily room URL
        call_id: Call identifier
        on_audio_from_daily: Callback for audio received from Daily room

    Returns:
        DailyBridge if successful, None otherwise
    """
    bridge = DailyBridge(
        room_url=room_url,
        call_id=call_id,
        on_audio_from_daily=on_audio_from_daily
    )

    if await bridge.join():
        return bridge

    return None
