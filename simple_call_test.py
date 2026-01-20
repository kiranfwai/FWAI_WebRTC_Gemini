"""
Simple Call Test - Just calls and prints what user says
No Gemini AI - just STT transcription
"""

import asyncio
import os
import io
import wave
import httpx
import aiohttp
from dotenv import load_dotenv

load_dotenv()

# Configuration
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

BASE_URL = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}"


async def transcribe_audio(audio_data: bytes, sample_rate: int = 16000) -> str:
    """Transcribe audio using Sarvam AI STT"""
    if not SARVAM_API_KEY:
        return "[No Sarvam API key]"

    try:
        # Convert PCM to WAV format
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        wav_buffer.seek(0)
        wav_bytes = wav_buffer.read()

        url = "https://api.sarvam.ai/speech-to-text"
        form_data = aiohttp.FormData()
        form_data.add_field('file', wav_bytes, filename='audio.wav', content_type='audio/wav')
        form_data.add_field('model', 'saarika:v2.5')
        form_data.add_field('language_code', 'hi-IN')

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                data=form_data,
                headers={"api-subscription-key": SARVAM_API_KEY},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("transcript", "[empty]")
                else:
                    return f"[STT error: {response.status}]"
    except Exception as e:
        return f"[Error: {str(e)[:50]}]"


async def make_simple_call(phone_number: str):
    """
    Make a call and just print what user says
    Uses the existing Docker services for WebRTC handling
    """
    print(f"\n{'='*60}")
    print(f"SIMPLE CALL TEST - Calling {phone_number}")
    print(f"{'='*60}\n")

    # Make call via main server
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:3000/make-call",
                json={"phoneNumber": phone_number, "callerName": "Test"},
                timeout=30.0
            )
            result = response.json()

            if result.get("success"):
                call_id = result.get("call_id")
                print(f"✓ Call initiated: {call_id}")
                print(f"\n>>> Listening for speech... (check Docker logs)")
                print(f">>> Run: docker-compose logs -f gemini-live")
                print(f"\n>>> Press Ctrl+C to stop\n")

                # Keep running until user stops
                while True:
                    await asyncio.sleep(1)
            else:
                print(f"✗ Call failed: {result.get('error')}")
                if 'details' in result:
                    print(f"  Details: {result['details']}")

    except KeyboardInterrupt:
        print("\n\nCall ended by user")
    except Exception as e:
        print(f"Error: {e}")


async def main():
    # Get phone number from command line or use default
    import sys
    phone = sys.argv[1] if len(sys.argv) > 1 else "919052034075"

    print("\nSimple WhatsApp Call Test")
    print("-------------------------")
    print(f"Phone Number ID: {PHONE_NUMBER_ID}")
    print(f"Target: {phone}")
    print(f"Sarvam STT: {'Configured' if SARVAM_API_KEY else 'Not configured'}")

    await make_simple_call(phone)


if __name__ == "__main__":
    asyncio.run(main())
