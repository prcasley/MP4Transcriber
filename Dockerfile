FROM python:3.12-slim

# Install ffmpeg and yt-dlp
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir yt-dlp requests flask flask-cors bgutil-ytdlp-pot-provider

WORKDIR /app
COPY transcribe_server.py .

EXPOSE 5111

CMD ["python", "transcribe_server.py"]
