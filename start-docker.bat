@echo off
REM Start WhatsApp Voice Calling with Gemini Live using Docker

echo ============================================
echo WhatsApp Voice Calling with Gemini Live
echo Docker Startup
echo ============================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Build and start containers
echo Building Docker images...
docker-compose build

echo.
echo Starting services...
docker-compose up -d

echo.
echo ============================================
echo Services started!
echo ============================================
echo.
echo Gemini Live Service: http://localhost:8003 (WebSocket)
echo Main Server:         http://localhost:3000
echo.
echo To make a call:
echo curl -X POST http://localhost:3000/make-call -H "Content-Type: application/json" -d "{\"phoneNumber\": \"919052034075\", \"contactName\": \"Test Customer\"}"
echo.
echo To view logs:
echo docker-compose logs -f
echo.
echo To stop:
echo docker-compose down
echo.
pause
