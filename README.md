# TranscribeHQ

Local audio/video transcription powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Drag-and-drop web UI, URL transcription (YouTube, Fathom, 1800+ sites), batch processing, multiple output formats.

## Quick Start

```bash
pip install -r requirements.txt
python transcribe_server.py
```

Then open the UI at **https://prcasley.github.io/MP4Transcriber** (or http://localhost:5111).

The frontend is hosted on GitHub Pages and connects to your local backend on port 5111.

## Features

- Drag-and-drop file upload (MP4, MP3, WAV, M4A, MOV, MKV, and more)
- **URL transcription** — paste a YouTube, Fathom, Vimeo, TikTok, or any video URL
- Supports **1,800+ websites** via [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- Fathom meeting transcripts with speaker names (via Fathom API)
- Auto-chunks files >25MB for Groq API compatibility
- 5 model sizes: tiny, base, small, medium, large-v3
- Voice Activity Detection (VAD) to skip silence
- Output formats: plain text, SRT subtitles, timestamped JSON
- Batch queue with real-time progress
- Auto-saves transcripts to `transcripts_output/`
- Runs 100% locally, no external APIs for file uploads

## Model Recommendations

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| tiny | Fastest | Low | Quick tests |
| base | Fast | Decent | General use |
| small | Moderate | Good | Most recordings |
| **medium** | **Slow** | **High** | **Old recordings, background noise** |
| large-v3 | Slowest | Best | Critical accuracy (needs GPU) |

## URL Transcription Setup

URL transcription requires a few extra dependencies:

```bash
pip install yt-dlp requests
```

You also need [ffmpeg](https://ffmpeg.org/) installed (for splitting large files):
- **Windows**: `winget install ffmpeg` or download from ffmpeg.org
- **Mac**: `brew install ffmpeg`
- **Linux**: `apt install ffmpeg`

Set your Groq API key for cloud transcription of URL content:

```bash
GROQ_API_KEY=your-key python transcribe_server.py
```

Optional: set `FATHOM_API_KEY` for direct Fathom meeting transcript access with speaker names.

### Supported Sites

YouTube, Vimeo, TikTok, Twitter/X, Loom, Twitch, Dailymotion, SoundCloud, Fathom, SharePoint, direct file links (.mp4/.mp3/etc.), and [1,800+ more](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md).

## GPU Support

Set `USE_GPU=1` for CUDA acceleration:

```bash
USE_GPU=1 python transcribe_server.py
```

Requires CUDA toolkit and cuDNN installed.
