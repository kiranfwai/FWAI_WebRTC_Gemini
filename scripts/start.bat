@echo off
REM Start WhatsApp Voice Calling with Gemini Live

REM Navigate to project root
cd /d "%~dp0.."

REM Activate virtual environment
call venv\Scripts\activate

REM Run the application
python run.py
