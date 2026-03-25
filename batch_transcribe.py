"""
Batch transcription CLI — process entire folders of audio/video files.
Designed for large volumes (100+ hours of lectures, recordings, etc.)

Usage:
    python batch_transcribe.py /path/to/videos
    python batch_transcribe.py /path/to/videos --model medium --language en
    python batch_transcribe.py /path/to/videos --resume  (skip already-transcribed files)

Output goes to transcripts_output/ by default (or --output /custom/path).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import timedelta

from faster_whisper import WhisperModel

SUPPORTED = {'.mp4', '.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm',
             '.avi', '.mov', '.mkv', '.wma', '.aac', '.opus'}


def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    h = int(td.total_seconds() // 3600)
    m = int((td.total_seconds() % 3600) // 60)
    s = int(td.total_seconds() % 60)
    ms = int((td.total_seconds() % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def find_files(input_path: Path, recursive: bool = True) -> list[Path]:
    files = []
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED:
            files.append(input_path)
    elif input_path.is_dir():
        pattern = '**/*' if recursive else '*'
        for f in sorted(input_path.glob(pattern)):
            if f.is_file() and f.suffix.lower() in SUPPORTED:
                files.append(f)
    return files


def transcribe_one(model: WhisperModel, file_path: Path, output_dir: Path,
                    language: str | None, file_num: int, total: int) -> dict:
    """Transcribe a single file and save outputs."""
    stem = file_path.stem
    safe_stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem)

    print(f"\n{'='*70}")
    print(f"  [{file_num}/{total}] {file_path.name}")
    print(f"  Size: {file_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"{'='*70}")

    start_time = time.time()

    segments, info = model.transcribe(
        str(file_path),
        language=language,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
        beam_size=5,
    )

    duration = info.duration or 0
    print(f"  Language: {info.language} ({info.language_probability:.0%} confidence)")
    print(f"  Audio duration: {format_duration(duration)}")
    print(f"  Transcribing", end="", flush=True)

    full_text_parts = []
    srt_parts = []
    segments_list = []
    seg_index = 0
    last_pct = -1

    for segment in segments:
        seg_index += 1
        text = segment.text.strip()
        if not text:
            continue

        # Progress dots
        if duration > 0:
            pct = int((segment.end / duration) * 100)
            if pct >= last_pct + 10:
                print(f" {pct}%", end="", flush=True)
                last_pct = pct

        full_text_parts.append(text)
        srt_parts.append(
            f"{seg_index}\n"
            f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
            f"{text}\n"
        )
        segments_list.append({
            "index": seg_index,
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": text,
        })

    elapsed = time.time() - start_time
    speed_ratio = duration / elapsed if elapsed > 0 else 0
    print(f"\n  Done in {format_duration(elapsed)} ({speed_ratio:.1f}x realtime)")
    print(f"  {seg_index} segments extracted")

    # Save outputs
    full_text = " ".join(full_text_parts)
    srt_text = "\n".join(srt_parts)
    json_result = {
        "file": file_path.name,
        "language": info.language,
        "duration_seconds": round(duration, 2),
        "transcription_seconds": round(elapsed, 2),
        "speed_ratio": round(speed_ratio, 1),
        "segments": segments_list,
    }

    (output_dir / f"{safe_stem}.txt").write_text(full_text, encoding="utf-8")
    (output_dir / f"{safe_stem}.srt").write_text(srt_text, encoding="utf-8")
    (output_dir / f"{safe_stem}.json").write_text(
        json.dumps(json_result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"  Saved: {safe_stem}.txt / .srt / .json")

    return {
        "file": file_path.name,
        "duration": duration,
        "elapsed": elapsed,
        "segments": seg_index,
        "language": info.language,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch transcribe audio/video files using Whisper"
    )
    parser.add_argument("input", help="File or folder to transcribe")
    parser.add_argument("--model", default="medium",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--language", default=None,
                        help="Language code (e.g., 'en'). Auto-detects if not set.")
    parser.add_argument("--output", default="transcripts_output",
                        help="Output directory (default: transcripts_output/)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip files that already have transcripts")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Don't search subdirectories")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU (CUDA) for transcription")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find files
    files = find_files(input_path, recursive=not args.no_recursive)
    if not files:
        print(f"No supported audio/video files found in: {input_path}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED))}")
        sys.exit(1)

    # Resume: skip already-transcribed
    if args.resume:
        existing = {f.stem for f in output_dir.glob("*.txt")}
        before = len(files)
        files = [f for f in files if f.stem not in existing]
        skipped = before - len(files)
        if skipped > 0:
            print(f"Resuming: skipping {skipped} already-transcribed file(s)")

    if not files:
        print("All files already transcribed. Nothing to do.")
        sys.exit(0)

    # Estimate total duration (rough, based on file sizes)
    total_size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)

    print(f"\n  TranscribeHQ — Batch Mode")
    print(f"  {'─'*40}")
    print(f"  Files:   {len(files)}")
    print(f"  Total:   {total_size_mb:.0f} MB")
    print(f"  Model:   {args.model}")
    print(f"  Device:  {'GPU (CUDA)' if args.gpu else 'CPU'}")
    print(f"  Output:  {output_dir.resolve()}")
    print(f"  {'─'*40}")

    # Load model
    print(f"\n  Loading {args.model} model...")
    device = "cuda" if args.gpu else "cpu"
    compute_type = "float16" if args.gpu else "int8"
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    print(f"  Model loaded on {device}")

    # Process files
    batch_start = time.time()
    results = []
    errors = []

    for i, file_path in enumerate(files, 1):
        try:
            result = transcribe_one(model, file_path, output_dir,
                                    args.language, i, len(files))
            results.append(result)
        except Exception as e:
            print(f"\n  ERROR on {file_path.name}: {e}")
            errors.append({"file": file_path.name, "error": str(e)})

    # Summary
    batch_elapsed = time.time() - batch_start
    total_audio = sum(r["duration"] for r in results)
    total_segments = sum(r["segments"] for r in results)

    print(f"\n{'='*70}")
    print(f"  BATCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Files processed:   {len(results)}/{len(files)}")
    if errors:
        print(f"  Errors:            {len(errors)}")
    print(f"  Total audio:       {format_duration(total_audio)}")
    print(f"  Total time:        {format_duration(batch_elapsed)}")
    print(f"  Avg speed:         {total_audio/batch_elapsed:.1f}x realtime" if batch_elapsed > 0 else "")
    print(f"  Total segments:    {total_segments}")
    print(f"  Output:            {output_dir.resolve()}")
    print(f"{'='*70}\n")

    # Save batch summary
    summary = {
        "batch_completed": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "device": device,
        "files_processed": len(results),
        "files_errored": len(errors),
        "total_audio_seconds": round(total_audio, 1),
        "total_processing_seconds": round(batch_elapsed, 1),
        "speed_ratio": round(total_audio / batch_elapsed, 1) if batch_elapsed > 0 else 0,
        "results": results,
        "errors": errors,
    }
    summary_path = output_dir / "_batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Summary saved: {summary_path}")

    if errors:
        print(f"\n  Failed files:")
        for err in errors:
            print(f"    - {err['file']}: {err['error']}")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
