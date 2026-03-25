"""
TranscribeHQ — Local transcription server powered by faster-whisper.
Handles file uploads, transcription with model selection, VAD filtering,
and outputs in txt/srt/json formats.

Usage:
    pip install faster-whisper flask flask-cors
    python transcribe_server.py
"""

import os
import json
import time
import uuid
import threading
from pathlib import Path
from datetime import timedelta

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from faster_whisper import WhisperModel

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("transcripts_output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job tracking
jobs: dict[str, dict] = {}
models_cache: dict[str, WhisperModel] = {}
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


if __name__ == "__main__":
    print("\n  TranscribeHQ Server")
    print("  Open http://localhost:5111 in your browser\n")
    app.run(host="0.0.0.0", port=5111, debug=False, threaded=True)
