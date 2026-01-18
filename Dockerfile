# WhatsApp Voice Calling with Gemini Live
# Docker image with all dependencies pre-installed

FROM python:3.11-slim

# Install system dependencies for aiortc and av
RUN apt-get update && apt-get install -y \
    # FFmpeg and audio/video libraries
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    # Audio libraries
    libopus-dev \
    libvpx-dev \
    # Build tools
    gcc \
    g++ \
    pkg-config \
    # SSL and networking
    libssl-dev \
    libffi-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    httpx==0.26.0 \
    websockets==12.0 \
    numpy==1.26.3 \
    scipy==1.11.4 \
    python-dotenv==1.0.0 \
    loguru==0.7.2 \
    av==12.3.0 \
    aiortc==1.9.0

# Copy application code
COPY . .

# Expose port
EXPOSE 3000

# Run the main server
CMD ["python", "main.py"]
