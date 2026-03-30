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
import time
import uuid
import subprocess
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

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

TRANSCRIBE_API_KEY = os.environ.get("TRANSCRIBE_API_KEY", "")


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
    elif any(d in url_lower for d in ["sharepoint.com", "onedrive.live.com", "1drv.ms"]):
        return "sharepoint"
    else:
        return "yt-dlp-generic"


def download_with_ytdlp(url: str, tmp_dir: str) -> str:
    """Download audio from any yt-dlp supported site."""
    output_template = os.path.join(tmp_dir, "audio.%(ext)s")
    # Check if PO token plugin is available
    plugin_check = subprocess.run(["yt-dlp", "--list-plugins"], capture_output=True, text=True, timeout=10)
    plugin_info = plugin_check.stdout + plugin_check.stderr

    cmd = [
        "yt-dlp", "--no-playlist",
        "-x", "--audio-format", "mp3", "--audio-quality", "5",
        "--max-filesize", "500m",
        "-o", output_template,
        "--no-warnings",
        "-v",  # verbose to see if PO token is being used
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed (plugins: {plugin_info[:200]}): {result.stderr[:500]}")

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
    resp = http_requests.get(url, stream=True, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download (HTTP {resp.status_code})")
    ext = Path(url.split("?")[0]).suffix or ".mp4"
    out_path = os.path.join(tmp_dir, f"audio{ext}")
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
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
                audio_path = download_with_ytdlp(url, tmp_dir)
            elif source_type == "direct":
                audio_path = download_direct(url, tmp_dir)
            elif source_type == "sharepoint":
                try:
                    audio_path = download_direct(url, tmp_dir)
                except Exception:
                    audio_path = download_with_ytdlp(url, tmp_dir)
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

    ffmpeg_ok = False
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        ffmpeg_ok = result.returncode == 0
    except Exception:
        pass

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
        "version": "2.1.0",
        "groq_key_set": bool(GROQ_API_KEY),
        "fathom_key_set": bool(FATHOM_API_KEY),
        "yt_dlp": ytdlp_version,
        "yt_dlp_plugins": plugins_info,
        "node": node_version,
        "ffmpeg": "installed" if ffmpeg_ok else "missing",
        "auth_required": bool(TRANSCRIBE_API_KEY),
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
