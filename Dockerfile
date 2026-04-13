FROM python:3.12-slim

# Install ffmpeg + Node.js (needed for PO token plugin's BotGuard challenge)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir requests flask flask-cors bgutil-ytdlp-pot-provider && \
    pip install --no-cache-dir "yt-dlp>=2026.3.17"

WORKDIR /app
COPY transcribe_server.py .

EXPOSE 5111

CMD ["python", "transcribe_server.py"]
