"""
TranscribeHQ — Local transcription server powered by faster-whisper.
Handles file uploads, URL transcription (YouTube, Fathom, Vimeo, 1800+ sites),
transcription with model selection, VAD filtering, and outputs in txt/srt/json formats.

Usage:
    pip install faster-whisper flask flask-cors yt-dlp requests
    python transcribe_server.py
"""

import os
import json
import shutil
import time
import uuid
import subprocess
import sys

# Auto-detect ffmpeg in common locations if not on PATH
_FFMPEG_DIRS = [
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "ffmpeg", "bin"),
    os.path.join(os.environ.get("ProgramFiles", ""), "ffmpeg", "bin"),
    r"C:\ffmpeg\bin",
]
for _d in _FFMPEG_DIRS:
    if os.path.isfile(os.path.join(_d, "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")):
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
        break
import tempfile
import threading
from pathlib import Path
from datetime import timedelta
from functools import wraps

import requests as http_requests
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # Running in cloud mode (Render) — local transcription disabled

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YT_TRANSCRIPT_API = True
except ImportError:
    YT_TRANSCRIPT_API = False

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

TRANSCRIBE_API_KEY = os.environ.get("TRANSCRIBE_API_KEY", "")

# Write YouTube cookies from env var to disk at startup (base64-encoded Netscape cookie file)
YOUTUBE_COOKIES_PATH = None
_yt_cookies_b64 = os.environ.get("YOUTUBE_COOKIES", "")
if _yt_cookies_b64:
    import base64
    _cookies_path = "/tmp/yt_cookies.txt"
    try:
        with open(_cookies_path, "wb") as _f:
            _f.write(base64.b64decode(_yt_cookies_b64))
        YOUTUBE_COOKIES_PATH = _cookies_path
        print(f"[YouTube] Cookies loaded from YOUTUBE_COOKIES env var → {_cookies_path}")
    except Exception as _e:
        print(f"[YouTube] Failed to write cookies: {_e}")


def require_api_key(f):
    """API key middleware. Skipped if TRANSCRIBE_API_KEY is not set."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not TRANSCRIBE_API_KEY:
            return f(*args, **kwargs)
        key = request.headers.get("X-API-Key", "") or request.args.get("api_key", "")
        if key != TRANSCRIBE_API_KEY:
            return jsonify({"error": "Unauthorized — invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("transcripts_output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job tracking
jobs: dict[str, dict] = {}
models_cache: dict = {}
model_lock = threading.Lock()

ALLOWED_EXTENSIONS = {
    ".mp4", ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm",
    ".avi", ".mov", ".mkv", ".wma", ".aac", ".opus",
}

def get_model(model_size: str) -> WhisperModel:
    """Load or retrieve cached Whisper model."""
    with model_lock:
        if model_size not in models_cache:
            print(f"Loading model: {model_size} (this may take a moment...)")
            # Use CPU by default; switch to "cuda" if you have a GPU
            device = "cuda" if os.environ.get("USE_GPU", "").lower() in ("1", "true", "yes") else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            models_cache[model_size] = WhisperModel(
                model_size, device=device, compute_type=compute_type
            )
            print(f"Model {model_size} loaded on {device}")
        return models_cache[model_size]


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm"""
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = int(td.total_seconds() % 60)
    millis = int((td.total_seconds() % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcribe_file(job_id: str, file_path: str, model_size: str, language: str | None):
    """Run transcription in a background thread."""
    job = jobs[job_id]
    job["status"] = "loading_model"

    try:
        model = get_model(model_size)
        job["status"] = "transcribing"
        job["progress"] = 0

        segments_list = []
        full_text_parts = []
        srt_parts = []
        seg_index = 0

        # Transcribe with VAD filter to skip silence
        segments, info = model.transcribe(
            file_path,
            language=language if language else None,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
            beam_size=5,
        )

        job["language"] = info.language
        job["language_probability"] = round(info.language_probability, 2)
        duration = info.duration or 1

        for segment in segments:
            seg_index += 1
            text = segment.text.strip()
            if not text:
                continue

            start = segment.start
            end = segment.end

            # Update progress
            job["progress"] = min(99, int((end / duration) * 100))

            # Plain text
            full_text_parts.append(text)

            # SRT format
            srt_parts.append(
                f"{seg_index}\n"
                f"{format_timestamp(start)} --> {format_timestamp(end)}\n"
                f"{text}\n"
            )

            # JSON segments
            segments_list.append({
                "index": seg_index,
                "start": round(start, 2),
                "end": round(end, 2),
                "text": text,
            })

        # Build results
        full_text = " ".join(full_text_parts)
        srt_text = "\n".join(srt_parts)
        json_result = {
            "file": os.path.basename(file_path),
            "language": info.language,
            "duration_seconds": round(duration, 2),
            "segments": segments_list,
        }

        # Save to output directory
        stem = Path(file_path).stem
        safe_stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem)

        (OUTPUT_DIR / f"{safe_stem}.txt").write_text(full_text, encoding="utf-8")
        (OUTPUT_DIR / f"{safe_stem}.srt").write_text(srt_text, encoding="utf-8")
        (OUTPUT_DIR / f"{safe_stem}.json").write_text(
            json.dumps(json_result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        job["status"] = "completed"
        job["progress"] = 100
        job["result"] = {
            "text": full_text,
            "srt": srt_text,
            "json": json_result,
            "segment_count": seg_index,
            "duration": round(duration, 2),
            "language": info.language,
            "output_dir": str(OUTPUT_DIR.resolve()),
        }

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"Transcription error for job {job_id}: {e}")

    finally:
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except OSError:
            pass


# ── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/upload", methods=["POST"])
@require_api_key
def upload():
    """Upload a file and start transcription."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    model_size = request.form.get("model", "base")
    language = request.form.get("language", "").strip() or None

    # Save uploaded file
    job_id = str(uuid.uuid4())[:8]
    safe_name = f"{job_id}_{file.filename}"
    file_path = str(UPLOAD_DIR / safe_name)
    file.save(file_path)

    # Create job
    jobs[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "model": model_size,
        "language": language,
        "status": "queued",
        "progress": 0,
        "created": time.time(),
        "result": None,
        "error": None,
    }

    # Start transcription in background
    thread = threading.Thread(
        target=transcribe_file,
        args=(job_id, file_path, model_size, language),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "filename": file.filename})


# ── Chunked upload for Power Automate / Office Scripts ──────────────

import base64

# In-memory storage for chunked upload sessions
upload_sessions: dict[str, dict] = {}


@app.route("/api/upload-base64", methods=["POST"])
@require_api_key
def upload_base64():
    """Upload a file as base64 JSON (for Office Scripts / Power Automate).

    For small files (< 4MB base64 string). For larger files use chunked upload.
    Body: { "filename": "video.mp4", "data": "<base64>", "model": "whisper-large-v3-turbo", "language": "en" }
    """
    data = request.get_json(force=True)
    filename = data.get("filename", "upload.mp4")
    b64_data = data.get("data", "")
    model_size = data.get("model", "whisper-large-v3-turbo")
    language = data.get("language", "").strip() or None

    if not b64_data:
        return jsonify({"error": "No data provided"}), 400

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    try:
        file_bytes = base64.b64decode(b64_data)
    except Exception:
        return jsonify({"error": "Invalid base64 data"}), 400

    job_id = str(uuid.uuid4())[:8]
    safe_name = f"{job_id}_{filename}"
    file_path = str(UPLOAD_DIR / safe_name)
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    jobs[job_id] = {
        "id": job_id,
        "filename": filename,
        "model": model_size,
        "language": language,
        "status": "queued",
        "progress": 0,
        "created": time.time(),
        "result": None,
        "error": None,
        "mode": "base64",
    }

    # Groq cloud models (whisper-*) use Groq API; local models use faster-whisper
    if model_size.startswith("whisper"):
        _start_groq_file_job(job_id, file_path, model_size, language)
    else:
        thread = threading.Thread(
            target=transcribe_file,
            args=(job_id, file_path, model_size, language),
            daemon=True,
        )
        thread.start()

    return jsonify({"job_id": job_id, "filename": filename})


@app.route("/api/upload-chunked/start", methods=["POST"])
@require_api_key
def chunked_start():
    """Start a chunked upload session.

    Body: { "filename": "video.mp4", "totalChunks": 10, "model": "whisper-large-v3-turbo", "language": "en" }
    Returns: { "session_id": "..." }
    """
    data = request.get_json(force=True)
    filename = data.get("filename", "upload.mp4")
    total_chunks = data.get("totalChunks", 1)
    model = data.get("model", "whisper-large-v3-turbo")
    language = data.get("language", "").strip() or None

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    session_id = str(uuid.uuid4())[:8]
    upload_sessions[session_id] = {
        "filename": filename,
        "total_chunks": total_chunks,
        "received_chunks": {},
        "model": model,
        "language": language,
        "created": time.time(),
    }

    print(f"[TranscribeHQ] Chunked upload session {session_id}: {filename} ({total_chunks} chunks)")
    return jsonify({"session_id": session_id, "filename": filename})


@app.route("/api/upload-chunked/chunk", methods=["POST"])
@require_api_key
def chunked_upload():
    """Upload a single chunk (base64).

    Body: { "session_id": "...", "chunkIndex": 0, "data": "<base64>" }
    """
    data = request.get_json(force=True)
    session_id = data.get("session_id", "")
    chunk_index = data.get("chunkIndex", 0)
    b64_data = data.get("data", "")

    session = upload_sessions.get(session_id)
    if not session:
        return jsonify({"error": "Invalid or expired session ID"}), 404

    try:
        chunk_bytes = base64.b64decode(b64_data)
    except Exception:
        return jsonify({"error": "Invalid base64 data"}), 400

    session["received_chunks"][chunk_index] = chunk_bytes
    received = len(session["received_chunks"])
    total = session["total_chunks"]

    print(f"[TranscribeHQ] Session {session_id}: chunk {chunk_index + 1}/{total} ({len(chunk_bytes)} bytes)")
    return jsonify({
        "session_id": session_id,
        "chunkIndex": chunk_index,
        "received": received,
        "total": total,
    })


@app.route("/api/upload-chunked/complete", methods=["POST"])
@require_api_key
def chunked_complete():
    """Finalize a chunked upload and start transcription.

    Body: { "session_id": "..." }
    Returns: { "job_id": "...", "filename": "..." }
    """
    data = request.get_json(force=True)
    session_id = data.get("session_id", "")

    session = upload_sessions.get(session_id)
    if not session:
        return jsonify({"error": "Invalid or expired session ID"}), 404

    received = len(session["received_chunks"])
    total = session["total_chunks"]
    if received < total:
        return jsonify({
            "error": f"Missing chunks: received {received}/{total}",
            "received": received,
            "total": total,
        }), 400

    # Reassemble file from chunks in order
    file_bytes = b""
    for i in range(total):
        chunk = session["received_chunks"].get(i)
        if chunk is None:
            return jsonify({"error": f"Missing chunk {i}"}), 400
        file_bytes += chunk

    filename = session["filename"]
    model = session["model"]
    language = session["language"]

    # Save assembled file
    job_id = str(uuid.uuid4())[:8]
    safe_name = f"{job_id}_{filename}"
    file_path = str(UPLOAD_DIR / safe_name)
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    file_size_mb = len(file_bytes) / (1024 * 1024)
    print(f"[TranscribeHQ] Session {session_id} complete: {filename} ({file_size_mb:.1f} MB) -> job {job_id}")

    # Clean up session
    del upload_sessions[session_id]

    # Create job and start transcription
    jobs[job_id] = {
        "id": job_id,
        "filename": filename,
        "model": model,
        "language": language,
        "status": "queued",
        "progress": 0,
        "created": time.time(),
        "result": None,
        "error": None,
        "mode": "chunked_upload",
        "file_size_mb": round(file_size_mb, 1),
    }

    _start_groq_file_job(job_id, file_path, model, language)

    return jsonify({"job_id": job_id, "filename": filename, "file_size_mb": round(file_size_mb, 1)})


def _start_groq_file_job(job_id: str, file_path: str, model: str, language: str | None):
    """Transcribe an uploaded file via Groq API (extract audio, chunk, send)."""
    def _run():
        job = jobs[job_id]
        try:
            job["status"] = "transcribing"
            job["progress"] = 5

            with tempfile.TemporaryDirectory() as tmp_dir:
                # Extract audio to MP3 (shrinks video files dramatically)
                audio_path = os.path.join(tmp_dir, "audio.mp3")
                job["progress"] = 10
                print(f"[TranscribeHQ] Job {job_id}: extracting audio...")

                ffmpeg_result = subprocess.run(
                    ["ffmpeg", "-y", "-i", file_path, "-vn", "-acodec", "libmp3lame",
                     "-q:a", "5", audio_path],
                    capture_output=True, text=True, timeout=600,
                )
                if ffmpeg_result.returncode != 0 or not os.path.exists(audio_path):
                    # If ffmpeg fails, try using the original file directly
                    audio_path = file_path

                audio_size = os.path.getsize(audio_path) / (1024 * 1024)
                print(f"[TranscribeHQ] Job {job_id}: audio extracted ({audio_size:.1f} MB)")

                # Chunk if needed
                job["progress"] = 20
                chunks = chunk_audio(audio_path, tmp_dir)

                # Transcribe each chunk via Groq
                all_segments = []
                full_text_parts = []
                detected_lang = language or "auto"

                for i, chunk_path in enumerate(chunks):
                    pct = 20 + int(((i + 1) / len(chunks)) * 75)
                    job["progress"] = min(95, pct)

                    groq_result = groq_transcribe(chunk_path, model, language)
                    text = groq_result.get("text", "").strip()
                    if text:
                        full_text_parts.append(text)

                    for seg in groq_result.get("segments", []):
                        all_segments.append({
                            "index": len(all_segments) + 1,
                            "start": round(seg.get("start", 0), 2),
                            "end": round(seg.get("end", 0), 2),
                            "text": (seg.get("text", "")).strip(),
                        })

                    if groq_result.get("language"):
                        detected_lang = groq_result["language"]

                    # Rate limit: 3s between chunks
                    if i < len(chunks) - 1:
                        time.sleep(3)

                # Build result
                full_text = " ".join(full_text_parts)
                srt_parts = []
                for idx, s in enumerate(all_segments, 1):
                    srt_parts.append(
                        f"{idx}\n"
                        f"{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}\n"
                        f"{s['text']}\n"
                    )

                duration = all_segments[-1]["end"] if all_segments else 0

                # Save to output dir
                stem = Path(file_path).stem
                safe_stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem)
                (OUTPUT_DIR / f"{safe_stem}.txt").write_text(full_text, encoding="utf-8")
                (OUTPUT_DIR / f"{safe_stem}.srt").write_text("\n".join(srt_parts), encoding="utf-8")

                job["status"] = "completed"
                job["progress"] = 100
                job["result"] = {
                    "text": full_text,
                    "srt": "\n".join(srt_parts),
                    "json": {"segments": all_segments, "language": detected_lang,
                             "duration_seconds": round(duration, 2)},
                    "segment_count": len(all_segments),
                    "duration": round(duration, 2),
                    "language": detected_lang,
                    "chunks_processed": len(chunks),
                }

        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            print(f"[TranscribeHQ] Groq file job error {job_id}: {e}")
        finally:
            try:
                os.remove(file_path)
            except OSError:
                pass

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


@app.route("/api/status/<job_id>")
def status(job_id: str):
    """Get job status."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/jobs")
def list_jobs():
    """List all jobs."""
    return jsonify(sorted(jobs.values(), key=lambda j: j["created"], reverse=True))


@app.route("/api/progress/<job_id>")
def progress_stream(job_id: str):
    """SSE stream for real-time progress updates."""
    def generate():
        while True:
            job = jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'not found'})}\n\n"
                break
            yield f"data: {json.dumps({'status': job['status'], 'progress': job['progress']})}\n\n"
            if job["status"] in ("completed", "error"):
                break
            time.sleep(0.5)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/download/<job_id>/<fmt>")
def download(job_id: str, fmt: str):
    """Download transcript in specified format."""
    job = jobs.get(job_id)
    if not job or job["status"] != "completed":
        return jsonify({"error": "Job not ready"}), 404

    result = job["result"]
    stem = Path(job["filename"]).stem
    safe_stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem)

    if fmt == "txt":
        return Response(
            result["text"],
            mimetype="text/plain",
            headers={"Content-Disposition": f"attachment; filename={safe_stem}.txt"},
        )
    elif fmt == "srt":
        return Response(
            result["srt"],
            mimetype="text/plain",
            headers={"Content-Disposition": f"attachment; filename={safe_stem}.srt"},
        )
    elif fmt == "json":
        return Response(
            json.dumps(result["json"], indent=2, ensure_ascii=False),
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment; filename={safe_stem}.json"},
        )
    else:
        return jsonify({"error": "Invalid format"}), 400


@app.route("/api/models")
def list_models():
    """List available Whisper models with descriptions."""
    return jsonify([
        {"id": "tiny", "name": "Tiny", "desc": "Fastest, lowest accuracy. Good for quick tests.", "size": "~75 MB"},
        {"id": "base", "name": "Base", "desc": "Fast with decent accuracy. Good starting point.", "size": "~140 MB"},
        {"id": "small", "name": "Small", "desc": "Good balance of speed and accuracy.", "size": "~460 MB"},
        {"id": "medium", "name": "Medium", "desc": "High accuracy, recommended for old recordings.", "size": "~1.5 GB"},
        {"id": "large-v3", "name": "Large v3", "desc": "Best accuracy. Needs GPU or patience on CPU.", "size": "~3 GB"},
    ])


# ── URL Transcription (YouTube, Fathom, Vimeo, 1800+ sites) ─────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
FATHOM_API_KEY = os.environ.get("FATHOM_API_KEY", "")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
MAX_GROQ_SIZE = 25 * 1024 * 1024  # 25MB Groq limit


def classify_url(url: str) -> str:
    """Classify a URL into a source type for routing."""
    url_lower = url.lower()

    if "fathom.video" in url_lower or "fathom.ai" in url_lower:
        return "fathom"
    elif any(d in url_lower for d in ["youtube.com", "youtu.be", "youtube-nocookie.com"]):
        return "youtube"
    elif any(d in url_lower for d in [
        "vimeo.com", "dailymotion.com", "twitch.tv", "tiktok.com",
        "twitter.com", "x.com", "facebook.com", "instagram.com",
        "reddit.com", "soundcloud.com", "bandcamp.com", "spotify.com",
        "rumble.com", "bitchute.com", "odysee.com", "loom.com",
    ]):
        return "yt-dlp"
    elif any(url_lower.split("?")[0].endswith(ext) for ext in [
        ".mp4", ".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac",
        ".avi", ".mkv", ".mov", ".wma", ".aac", ".opus"
    ]):
        return "direct"
    elif any(d in url_lower for d in [
        "sharepoint.com", "onedrive.live.com", "1drv.ms",
        "my.sharepoint.com", "sharepoint.com/:v:",
        "sharepoint.com/:u:", "sharepoint.com/:a:",
    ]):
        return "sharepoint"
    else:
        return "yt-dlp-generic"


def fetch_youtube_transcript(url: str) -> dict | None:
    """Fetch YouTube captions directly — no audio download, no bot detection."""
    if not YT_TRANSCRIPT_API:
        return None
    import re
    match = re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})', url)
    if not match:
        return None
    video_id = match.group(1)
    try:
        # Pass cookies via a requests Session so Render's blocked IP can authenticate
        session = None
        if YOUTUBE_COOKIES_PATH:
            import requests as _requests
            from http.cookiejar import MozillaCookieJar
            session = _requests.Session()
            jar = MozillaCookieJar(YOUTUBE_COOKIES_PATH)
            jar.load(ignore_discard=True, ignore_expires=True)
            session.cookies = jar
        api = YouTubeTranscriptApi(http_client=session)
        # Try English first, fall back to any available language
        try:
            fetched = api.fetch(video_id, languages=['en'])
        except Exception:
            transcript_list = api.list(video_id)
            first = next(iter(transcript_list))
            fetched = api.fetch(video_id, languages=[first.language_code])
        snippets = fetched.snippets
        full_text = " ".join(s.text.replace('\n', ' ') for s in snippets)
        srt_parts = []
        for i, s in enumerate(snippets, 1):
            end = s.start + s.duration
            srt_parts.append(
                f"{i}\n{format_timestamp(s.start)} --> {format_timestamp(end)}\n{s.text}\n"
            )
        return {
            "text": full_text,
            "srt": "\n".join(srt_parts),
            "language": fetched.language_code if hasattr(fetched, 'language_code') else "en",
        }
    except Exception as e:
        print(f"[YouTube transcript API] failed: {e}")
        return None


def download_with_ytdlp(url: str, tmp_dir: str) -> str:
    """Download audio from any yt-dlp supported site."""
    output_template = os.path.join(tmp_dir, "audio.%(ext)s")
    cmd = [
        "yt-dlp", "--no-playlist",
        "-x", "--audio-format", "mp3", "--audio-quality", "5",
        "--format", "bestaudio/best",
        "--max-filesize", "500m",
        "-o", output_template,
        "--no-warnings", "--quiet",
    ]
    if YOUTUBE_COOKIES_PATH:
        cmd += ["--cookies", YOUTUBE_COOKIES_PATH]
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[:500]}")

    for f in Path(tmp_dir).iterdir():
        if f.suffix in [".mp3", ".m4a", ".wav", ".ogg", ".webm", ".opus", ".flac"]:
            return str(f)
    raise RuntimeError("yt-dlp completed but no audio file found")


def download_from_fathom(url: str, tmp_dir: str) -> dict:
    """Handle Fathom URLs — try API first, then fall back to yt-dlp."""
    if FATHOM_API_KEY:
        recording_id = url.rstrip("/").split("/")[-1]
        try:
            resp = http_requests.get(
                "https://api.fathom.ai/external/v1/meetings",
                headers={"X-Api-Key": FATHOM_API_KEY},
                params={"include_transcript": "true"},
                timeout=30,
            )
            if resp.status_code == 200:
                for meeting in resp.json().get("items", []):
                    if recording_id in meeting.get("url", "") or recording_id in meeting.get("share_url", ""):
                        parts = []
                        for seg in meeting.get("transcript", []):
                            speaker = seg.get("speaker", {}).get("display_name", "Unknown")
                            text = seg.get("text", "")
                            ts = seg.get("timestamp", "")
                            parts.append(f"[{ts}] {speaker}: {text}")
                        return {
                            "source": "fathom_api",
                            "transcript": "\n".join(parts),
                            "title": meeting.get("title", "Fathom Meeting"),
                        }
        except Exception as e:
            print(f"[TranscribeHQ] Fathom API failed, falling back: {e}")

    audio_path = download_with_ytdlp(url, tmp_dir)
    return {"source": "fathom_download", "audio_path": audio_path}


def download_direct(url: str, tmp_dir: str) -> str:
    """Download a direct file URL."""
    resp = http_requests.get(url, stream=True, timeout=120, allow_redirects=True)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download (HTTP {resp.status_code})")
    # Detect content type to avoid saving HTML pages as video
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        raise RuntimeError("URL returned HTML instead of a media file — may require authentication")
    ext = Path(url.split("?")[0]).suffix or ".mp4"
    out_path = os.path.join(tmp_dir, f"audio{ext}")
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return out_path


def resolve_sharepoint_download_url(sharing_url: str) -> str:
    """Convert a SharePoint/OneDrive sharing link to a direct download URL.

    SharePoint anonymous sharing links (e.g. https://tenant.sharepoint.com/:v:/s/site/Exxxx)
    redirect to a viewer page. This converts them to direct download URLs using the
    SharePoint API's download endpoint.
    """
    import base64

    # Method 1: SharePoint sharing URL API encoding trick
    # Encode the sharing URL to a token that SharePoint's API understands
    # Base64url-encode the sharing URL, then use the shares endpoint
    encoded = base64.urlsafe_b64encode(sharing_url.encode()).decode().rstrip("=")
    share_token = "u!" + encoded

    # Try the Graph-style driveItem download via the shares endpoint
    # This works for anonymous links without authentication
    api_url = f"https://api.onedrive.com/v1.0/shares/{share_token}/root/content"

    # First do a HEAD request to get the redirect URL (actual file location)
    resp = http_requests.head(api_url, allow_redirects=False, timeout=30)
    if resp.status_code in (301, 302):
        return resp.headers.get("Location", sharing_url)

    # Method 2: Modify the SharePoint URL to force download
    # For URLs like /:v:/s/site/Exxxx, append &download=1
    if "sharepoint.com" in sharing_url:
        sep = "&" if "?" in sharing_url else "?"
        return sharing_url + sep + "download=1"

    return sharing_url


def download_from_sharepoint(url: str, tmp_dir: str) -> str:
    """Download a file from a SharePoint/OneDrive sharing link."""
    print(f"[TranscribeHQ] Resolving SharePoint sharing link...")

    # Try to resolve to a direct download URL
    try:
        download_url = resolve_sharepoint_download_url(url)
        print(f"[TranscribeHQ] Resolved to download URL")
    except Exception as e:
        print(f"[TranscribeHQ] Could not resolve sharing link: {e}")
        download_url = url

    # Download with redirect following
    resp = http_requests.get(download_url, stream=True, timeout=300, allow_redirects=True)
    if resp.status_code != 200:
        raise RuntimeError(f"SharePoint download failed (HTTP {resp.status_code})")

    # Check we got actual media, not an HTML viewer page
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        # Last resort: try appending download=1 to original URL
        sep = "&" if "?" in url else "?"
        resp = http_requests.get(url + sep + "download=1", stream=True, timeout=300, allow_redirects=True)
        if resp.status_code != 200 or "text/html" in resp.headers.get("Content-Type", ""):
            raise RuntimeError(
                "SharePoint returned a viewer page instead of the file. "
                "The sharing link may require sign-in. Ensure the link is set to "
                "'Anyone with the link' (anonymous access)."
            )

    # Determine extension from Content-Disposition or URL
    ext = ".mp4"
    cd = resp.headers.get("Content-Disposition", "")
    if "filename=" in cd:
        fname = cd.split("filename=")[-1].strip('" ')
        ext = Path(fname).suffix or ext
    elif Path(url.split("?")[0]).suffix:
        ext = Path(url.split("?")[0]).suffix

    out_path = os.path.join(tmp_dir, f"sharepoint_audio{ext}")
    total_bytes = 0
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            total_bytes += len(chunk)

    print(f"[TranscribeHQ] Downloaded {total_bytes / (1024*1024):.1f} MB from SharePoint")

    if total_bytes < 1000:
        raise RuntimeError("Downloaded file is too small — likely not a valid media file")

    return out_path


def chunk_audio(file_path: str, tmp_dir: str, chunk_mb: int = 20) -> list[str]:
    """Split large audio into chunks using ffmpeg if over Groq's 25MB limit."""
    file_size = os.path.getsize(file_path)
    if file_size <= MAX_GROQ_SIZE:
        return [file_path]

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        capture_output=True, text=True, timeout=30,
    )
    duration = float(probe.stdout.strip())
    num_chunks = max(2, int(file_size / (chunk_mb * 1024 * 1024)) + 1)
    chunk_duration = duration / num_chunks

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_duration
        chunk_path = os.path.join(tmp_dir, f"chunk_{i:03d}.mp3")
        subprocess.run([
            "ffmpeg", "-y", "-i", file_path,
            "-ss", str(start), "-t", str(chunk_duration),
            "-vn", "-acodec", "libmp3lame", "-q:a", "5",
            chunk_path
        ], capture_output=True, timeout=120)
        if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
            chunks.append(chunk_path)
    return chunks


def groq_transcribe(audio_path: str, model: str = "whisper-large-v3-turbo",
                     language: str | None = None) -> dict:
    """Send audio to Groq's Whisper API, return verbose JSON response."""
    with open(audio_path, "rb") as f:
        files = {"file": (Path(audio_path).name, f)}
        data = {"model": model, "response_format": "verbose_json", "temperature": "0"}
        if language:
            data["language"] = language

        resp = http_requests.post(
            GROQ_ENDPOINT,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files=files, data=data, timeout=300,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Groq API error ({resp.status_code}): {resp.text[:300]}")
    return resp.json()


def transcribe_url_job(job_id: str, url: str, groq_model: str, language: str | None):
    """Background thread: download audio from URL, transcribe via Groq."""
    job = jobs[job_id]
    source_type = classify_url(url)
    job["source_type"] = source_type

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # ── Download phase ──
            job["status"] = "downloading"
            job["progress"] = 5

            if source_type == "fathom":
                result = download_from_fathom(url, tmp_dir)
                if result.get("source") == "fathom_api":
                    # Got transcript directly from Fathom API
                    job["status"] = "completed"
                    job["progress"] = 100
                    job["result"] = {
                        "text": result["transcript"],
                        "srt": "",
                        "json": {"segments": [], "source": "fathom_api"},
                        "segment_count": 0,
                        "duration": 0,
                        "language": "en",
                        "title": result.get("title", ""),
                        "note": "Transcript from Fathom API (includes speaker names)",
                    }
                    return
                audio_path = result["audio_path"]
            elif source_type in ("youtube", "yt-dlp", "yt-dlp-generic"):
                if source_type == "youtube":
                    yt_transcript = fetch_youtube_transcript(url)
                    if yt_transcript:
                        job["status"] = "completed"
                        job["progress"] = 100
                        job["result"] = {
                            "text": yt_transcript["text"],
                            "srt": yt_transcript["srt"],
                            "json": {"segments": [], "source": "youtube_transcript_api"},
                            "segment_count": 0,
                            "duration": 0,
                            "language": yt_transcript["language"],
                            "note": "Transcript fetched directly from YouTube captions",
                        }
                        return
                audio_path = download_with_ytdlp(url, tmp_dir)
            elif source_type == "direct":
                audio_path = download_direct(url, tmp_dir)
            elif source_type == "sharepoint":
                audio_path = download_from_sharepoint(url, tmp_dir)
            else:
                raise RuntimeError(f"Unsupported URL type: {source_type}")

            # ── Chunk if needed ──
            job["status"] = "transcribing"
            job["progress"] = 30
            chunks = chunk_audio(audio_path, tmp_dir)

            # ── Transcribe each chunk via Groq ──
            all_segments = []
            full_text_parts = []
            detected_lang = language or "auto"

            for i, chunk_path in enumerate(chunks):
                pct = 30 + int(((i + 1) / len(chunks)) * 65)
                job["progress"] = min(95, pct)

                groq_result = groq_transcribe(chunk_path, groq_model, language)
                text = groq_result.get("text", "").strip()
                if text:
                    full_text_parts.append(text)

                for seg in groq_result.get("segments", []):
                    all_segments.append({
                        "index": len(all_segments) + 1,
                        "start": round(seg.get("start", 0), 2),
                        "end": round(seg.get("end", 0), 2),
                        "text": (seg.get("text", "")).strip(),
                    })

                if groq_result.get("language"):
                    detected_lang = groq_result["language"]

            # ── Build result ──
            full_text = " ".join(full_text_parts)
            srt_parts = []
            for i, s in enumerate(all_segments, 1):
                srt_parts.append(
                    f"{i}\n"
                    f"{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}\n"
                    f"{s['text']}\n"
                )

            duration = all_segments[-1]["end"] if all_segments else 0
            json_result = {
                "file": url,
                "language": detected_lang,
                "duration_seconds": round(duration, 2),
                "segments": all_segments,
            }

            # Save to output directory
            safe_stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in job["filename"][:80])
            (OUTPUT_DIR / f"{safe_stem}.txt").write_text(full_text, encoding="utf-8")
            (OUTPUT_DIR / f"{safe_stem}.srt").write_text("\n".join(srt_parts), encoding="utf-8")
            (OUTPUT_DIR / f"{safe_stem}.json").write_text(
                json.dumps(json_result, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            job["status"] = "completed"
            job["progress"] = 100
            job["result"] = {
                "text": full_text,
                "srt": "\n".join(srt_parts),
                "json": json_result,
                "segment_count": len(all_segments),
                "duration": round(duration, 2),
                "language": detected_lang,
                "chunks_processed": len(chunks),
            }

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"[TranscribeHQ] URL transcription error for {job_id}: {e}")


@app.route("/api/transcribe-url", methods=["POST"])
@require_api_key
def transcribe_url():
    """Accept a URL (YouTube, Fathom, Vimeo, etc.), download and transcribe."""
    data = request.get_json(force=True)
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    if not GROQ_API_KEY:
        return jsonify({"error": "GROQ_API_KEY not configured on server"}), 500

    groq_model = data.get("model", "whisper-large-v3-turbo")
    language = data.get("language", "").strip() or None

    # Extract a display name from the URL
    source_type = classify_url(url)
    display_name = url.split("/")[-1].split("?")[0] or url[:60]
    if source_type == "youtube":
        # Try to get video ID for display
        for param in url.split("?")[-1].split("&"):
            if param.startswith("v="):
                display_name = f"youtube_{param[2:]}"
                break

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "id": job_id,
        "filename": display_name,
        "model": groq_model,
        "language": language,
        "status": "queued",
        "progress": 0,
        "created": time.time(),
        "result": None,
        "error": None,
        "source_url": url,
        "source_type": source_type,
        "mode": "url",
    }

    thread = threading.Thread(
        target=transcribe_url_job,
        args=(job_id, url, groq_model, language),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "filename": display_name, "source_type": source_type})


@app.route("/api/supported-sites")
def supported_sites():
    """List supported URL sources."""
    return jsonify({
        "sources": [
            {"type": "youtube", "examples": ["youtube.com/watch?v=...", "youtu.be/..."]},
            {"type": "fathom", "examples": ["fathom.video/share/..."], "note": "Set FATHOM_API_KEY for speaker names"},
            {"type": "video_platforms", "examples": ["Vimeo, TikTok, Twitter/X, Loom, Twitch, Dailymotion, 1800+ sites"]},
            {"type": "direct_files", "examples": ["any URL ending in .mp4, .mp3, .wav, etc."]},
            {"type": "sharepoint", "examples": ["sharepoint.com/...", "onedrive.live.com/..."]},
        ],
    })


@app.route("/api/health")
def health():
    """Health check with dependency status."""
    ytdlp_version = "not installed"
    ytdlp_ok = False
    try:
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            ytdlp_ok = True
            ytdlp_version = result.stdout.strip()
    except Exception:
        pass

    ffmpeg_ok = bool(shutil.which("ffmpeg"))

    # Check yt-dlp plugins
    plugins_info = ""
    try:
        result = subprocess.run(["yt-dlp", "--list-plugins"], capture_output=True, text=True, timeout=5)
        plugins_info = (result.stdout + result.stderr).strip()[:300]
    except Exception:
        plugins_info = "could not check"

    # Check Node.js
    node_version = "not installed"
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            node_version = result.stdout.strip()
    except Exception:
        pass

    return jsonify({
        "status": "ok" if (ytdlp_ok and GROQ_API_KEY) else "degraded",
        "version": "2.1.1",
        "groq_key_set": bool(GROQ_API_KEY),
        "fathom_key_set": bool(FATHOM_API_KEY),
        "yt_dlp": ytdlp_version,
        "yt_dlp_plugins": plugins_info,
        "node": node_version,
        "ffmpeg": "installed" if ffmpeg_ok else "missing",
        "auth_required": bool(TRANSCRIBE_API_KEY),
        "yt_transcript_api": YT_TRANSCRIPT_API,
    })


@app.route("/api/openapi.json")
def openapi_spec():
    """OpenAPI 3.0 spec for Copilot Studio custom connector import."""
    host = request.host_url.rstrip("/")
    return jsonify({
        "openapi": "3.0.0",
        "info": {
            "title": "TranscribeHQ Universal",
            "description": "Transcribe audio/video from any URL — YouTube, Fathom, Vimeo, direct links, and 1800+ sites",
            "version": "2.1.0",
        },
        "servers": [{"url": host}],
        "paths": {
            "/api/transcribe-url": {
                "post": {
                    "operationId": "transcribeFromUrl",
                    "summary": "Transcribe audio/video from any URL",
                    "description": "Accepts YouTube, Fathom, Vimeo, direct file links, SharePoint, and 1800+ other sites. Returns a job ID for polling status.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["url"],
                                    "properties": {
                                        "url": {"type": "string", "description": "URL of the audio/video"},
                                        "model": {"type": "string", "enum": ["whisper-large-v3-turbo", "whisper-large-v3"], "default": "whisper-large-v3-turbo"},
                                        "language": {"type": "string", "description": "ISO language code or omit for auto-detect"},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Job created — poll /api/status/{job_id} for result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "job_id": {"type": "string"},
                                            "filename": {"type": "string"},
                                            "source_type": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        }
                    },
                    "security": [{"apiKeyAuth": []}],
                }
            },
            "/api/status/{job_id}": {
                "get": {
                    "operationId": "getJobStatus",
                    "summary": "Get transcription job status and result",
                    "parameters": [{
                        "name": "job_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }],
                    "responses": {
                        "200": {
                            "description": "Job status with transcript when completed",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string", "enum": ["queued", "downloading", "transcribing", "completed", "error"]},
                                            "progress": {"type": "integer"},
                                            "result": {
                                                "type": "object",
                                                "properties": {
                                                    "text": {"type": "string"},
                                                    "srt": {"type": "string"},
                                                    "segment_count": {"type": "integer"},
                                                    "language": {"type": "string"},
                                                    "duration": {"type": "number"},
                                                },
                                            },
                                            "error": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/api/supported-sites": {
                "get": {
                    "operationId": "listSupportedSites",
                    "summary": "List supported source types",
                    "responses": {"200": {"description": "Supported sources"}},
                }
            },
            "/api/health": {
                "get": {
                    "operationId": "healthCheck",
                    "summary": "Server status and dependency check",
                    "responses": {"200": {"description": "Health status"}},
                }
            },
        },
        "components": {
            "securitySchemes": {
                "apiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
            }
        },
    })


if __name__ == "__main__":
    print("\n  TranscribeHQ Server v2.1")
    print("  http://localhost:5111")
    print(f"  Groq API:  {'configured' if GROQ_API_KEY else 'NOT SET — URL transcription disabled'}")
    print(f"  Fathom:    {'configured' if FATHOM_API_KEY else 'not set (optional)'}")
    print(f"  Auth:      {'API key required' if TRANSCRIBE_API_KEY else 'open (set TRANSCRIBE_API_KEY to secure)'}")
    print(f"  OpenAPI:   http://localhost:5111/api/openapi.json")
    print()
    port = int(os.environ.get("PORT", 5111))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
