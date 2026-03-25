# TranscribeHQ

Local audio/video transcription powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Drag-and-drop web UI, batch processing, multiple output formats.

## Quick Start

```bash
pip install -r requirements.txt
python transcribe_server.py
```

Then open the UI at **https://prcasley.github.io/MP4Transcriber** (or http://localhost:5111).

The frontend is hosted on GitHub Pages and connects to your local backend on port 5111.

## Features

- Drag-and-drop file upload (MP4, MP3, WAV, M4A, MOV, MKV, and more)
- 5 model sizes: tiny, base, small, medium, large-v3
- Voice Activity Detection (VAD) to skip silence
- Output formats: plain text, SRT subtitles, timestamped JSON
- Batch queue with real-time progress
- Auto-saves transcripts to `transcripts_output/`
- Runs 100% locally, no external APIs

## Model Recommendations

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| tiny | Fastest | Low | Quick tests |
| base | Fast | Decent | General use |
| small | Moderate | Good | Most recordings |
| **medium** | **Slow** | **High** | **Old recordings, background noise** |
| large-v3 | Slowest | Best | Critical accuracy (needs GPU) |

## GPU Support

Set `USE_GPU=1` for CUDA acceleration:

```bash
USE_GPU=1 python transcribe_server.py
```

Requires CUDA toolkit and cuDNN installed.
