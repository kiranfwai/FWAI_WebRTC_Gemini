#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Transcription Script with Speaker Labels
Processes recordings older than 30 minutes, generates speaker-labeled transcripts.

Usage:
    python3 scripts/batch_transcribe.py

Schedule with cron (every hour):
    0 * * * * cd /home/kiran/FWAI_WebRTC_Gemini/FWAI_WebRTC_Gemini && /usr/bin/python3 scripts/batch_transcribe.py >> logs/transcribe.log 2>&1
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Directories
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
ARCHIVE_DIR = PROJECT_ROOT / "recordings_transcribed"

# Create directories if they don't exist
RECORDINGS_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR.mkdir(exist_ok=True)
ARCHIVE_DIR.mkdir(exist_ok=True)

# Settings
MIN_AGE_MINUTES = 30  # Only process files older than this
WHISPER_MODEL = "tiny"  # tiny, base, small, medium, large


def get_file_age_minutes(file_path: Path) -> float:
    """Get file age in minutes"""
    mtime = file_path.stat().st_mtime
    age_seconds = time.time() - mtime
    return age_seconds / 60


def transcribe_with_whisper(audio_file: Path) -> str:
    """Transcribe audio file using Whisper"""
    try:
        import whisper
        print(f"    Transcribing {audio_file.name}...")
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(str(audio_file))
        return result["text"].strip()
    except ImportError:
        print("    ERROR: Whisper not installed. Run: pip install openai-whisper")
        return None
    except Exception as e:
        print(f"    ERROR: Transcription failed: {e}")
        return None


def compress_to_mp3(wav_file: Path) -> Path:
    """Compress WAV to MP3 using ffmpeg"""
    mp3_file = wav_file.with_suffix('.mp3')
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', str(wav_file), '-b:a', '32k', str(mp3_file)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return mp3_file
        else:
            print(f"    WARNING: ffmpeg failed: {result.stderr}")
            return wav_file
    except FileNotFoundError:
        print("    WARNING: ffmpeg not found, keeping WAV format")
        return wav_file


def get_call_uuids():
    """Get unique call UUIDs from recordings directory"""
    uuids = set()
    for f in RECORDINGS_DIR.glob("*.wav"):
        # Extract UUID (remove _user or _agent suffix)
        name = f.stem
        if name.endswith("_user"):
            uuid = name[:-5]
        elif name.endswith("_agent"):
            uuid = name[:-6]
        else:
            uuid = name
        uuids.add(uuid)
    return uuids


def process_recording(call_uuid: str):
    """Process a single call recording: transcribe USER and AGENT separately"""
    print(f"\nProcessing call: {call_uuid}")

    # Find files for this call
    combined_file = RECORDINGS_DIR / f"{call_uuid}.wav"
    user_file = RECORDINGS_DIR / f"{call_uuid}_user.wav"
    agent_file = RECORDINGS_DIR / f"{call_uuid}_agent.wav"

    # Check if main file exists and is old enough
    if not combined_file.exists():
        print(f"  Skipping - no combined file found")
        return False

    age = get_file_age_minutes(combined_file)
    if age < MIN_AGE_MINUTES:
        print(f"  Skipping - too recent ({age:.1f} min old)")
        return False

    print(f"  Age: {age:.1f} minutes")

    # Transcribe each speaker
    user_text = None
    agent_text = None

    if user_file.exists():
        print(f"  Transcribing USER audio...")
        user_text = transcribe_with_whisper(user_file)
        if user_text:
            print(f"    USER: {user_text[:100]}...")
    else:
        print(f"  No separate user audio file")

    if agent_file.exists():
        print(f"  Transcribing AGENT audio...")
        agent_text = transcribe_with_whisper(agent_file)
        if agent_text:
            print(f"    AGENT: {agent_text[:100]}...")
    else:
        print(f"  No separate agent audio file")

    # If no separate files, transcribe combined (fallback)
    if not user_text and not agent_text:
        print(f"  Transcribing combined audio (no speaker separation)...")
        combined_text = transcribe_with_whisper(combined_file)
        if combined_text:
            user_text = None
            agent_text = None
            # Save without speaker labels
            transcript_file = TRANSCRIPTS_DIR / f"{call_uuid}.txt"
            with open(transcript_file, "a") as f:
                f.write(f"\n--- WHISPER TRANSCRIPTION ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ---\n")
                f.write(f"{combined_text}\n")
            print(f"  Transcript saved (no speaker labels): {transcript_file.name}")
    else:
        # Save speaker-labeled transcript
        transcript_file = TRANSCRIPTS_DIR / f"{call_uuid}.txt"
        with open(transcript_file, "a") as f:
            f.write(f"\n--- WHISPER TRANSCRIPTION ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ---\n")
            if agent_text:
                f.write(f"\n[AGENT]:\n{agent_text}\n")
            if user_text:
                f.write(f"\n[USER]:\n{user_text}\n")
        print(f"  Transcript saved with speaker labels: {transcript_file.name}")

    # Compress and archive files
    files_to_archive = []

    # Combined file
    if combined_file.exists():
        compressed = compress_to_mp3(combined_file)
        if compressed != combined_file:
            combined_file.unlink()
            combined_file = compressed
        files_to_archive.append(combined_file)

    # User file - just delete (we have the transcript)
    if user_file.exists():
        user_file.unlink()
        print(f"  Deleted: {user_file.name}")

    # Agent file - just delete (we have the transcript)
    if agent_file.exists():
        agent_file.unlink()
        print(f"  Deleted: {agent_file.name}")

    # Move combined to archive
    for f in files_to_archive:
        archive_path = ARCHIVE_DIR / f.name
        shutil.move(str(f), str(archive_path))
        print(f"  Archived: {archive_path.name}")

    return True


def main():
    print(f"=" * 60)
    print(f"Batch Transcription - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 60)
    print(f"Recordings dir: {RECORDINGS_DIR}")
    print(f"Transcripts dir: {TRANSCRIPTS_DIR}")
    print(f"Archive dir: {ARCHIVE_DIR}")
    print(f"Min age: {MIN_AGE_MINUTES} minutes")
    print(f"Whisper model: {WHISPER_MODEL}")

    # Get unique call UUIDs
    uuids = get_call_uuids()

    if not uuids:
        print(f"\nNo recordings found in {RECORDINGS_DIR}")
        return

    print(f"\nFound {len(uuids)} unique call(s)")

    # Process each call
    processed = 0
    failed = 0
    skipped = 0

    for uuid in sorted(uuids):
        try:
            result = process_recording(uuid)
            if result:
                processed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n" + "=" * 60)
    print(f"Complete: {processed} processed, {skipped} skipped, {failed} failed")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
