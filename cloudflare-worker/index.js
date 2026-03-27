/**
 * TranscribeHQ — Cloudflare Worker
 *
 * Accepts any URL, handles it intelligently:
 *   - YouTube: extracts built-in captions (instant, no download)
 *   - Fathom: pulls transcript via Fathom API (with speaker names)
 *   - Direct files: fetches audio, sends to Groq Whisper
 *
 * Secrets (set via `npx wrangler secret put`):
 *   GROQ_API_KEY    — required for direct file transcription
 *   FATHOM_API_KEY  — optional, for Fathom meeting transcripts
 *
 * Deploy:
 *   cd cloudflare-worker
 *   npx wrangler deploy
 */

const GROQ_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions";
const MAX_GROQ_SIZE = 25 * 1024 * 1024; // 25MB

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS_HEADERS });
    }

    const url = new URL(request.url);

    if (url.pathname === "/transcribe-url" && request.method === "POST") {
      return handleTranscribe(request, env);
    }

    if (url.pathname === "/health") {
      return jsonResponse({
        status: "ok",
        groq_key_set: !!env.GROQ_API_KEY,
        fathom_key_set: !!env.FATHOM_API_KEY,
      });
    }

    return jsonResponse({ error: "Not found. POST to /transcribe-url" }, 404);
  },
};

// ── Main handler ─────────────────────────────────────────────

async function handleTranscribe(request, env) {
  let body;
  try {
    body = await request.json();
  } catch {
    return jsonResponse({ error: "Invalid JSON body" }, 400);
  }

  const sourceUrl = (body.url || "").trim();
  if (!sourceUrl) {
    return jsonResponse({ error: "No URL provided" }, 400);
  }

  const model = body.model || "whisper-large-v3-turbo";
  const language = body.language || undefined;
  const sourceType = classifyUrl(sourceUrl);

  try {
    // ── YouTube: extract built-in captions ──
    if (sourceType === "youtube") {
      const result = await getYouTubeTranscript(sourceUrl, language);
      return jsonResponse({
        transcript: result.transcript,
        segments: result.segments,
        language: result.language,
        duration: result.duration,
        source_type: "youtube",
        title: result.title,
        note: "Transcript extracted from YouTube captions",
      });
    }

    // ── Fathom: try API ──
    if (sourceType === "fathom") {
      if (env.FATHOM_API_KEY) {
        const fathomResult = await tryFathomApi(sourceUrl, env.FATHOM_API_KEY);
        if (fathomResult) {
          return jsonResponse({
            transcript: fathomResult.transcript,
            source_type: "fathom",
            title: fathomResult.title,
            note: "Transcript from Fathom API (includes speaker names)",
          });
        }
      }
      throw new Error(
        "Fathom transcription requires a Fathom API key. " +
        "Ask your admin to set FATHOM_API_KEY on the worker."
      );
    }

    // ── Unsupported platforms ──
    if (sourceType === "platform") {
      const domain = new URL(sourceUrl).hostname.replace("www.", "");
      throw new Error(
        `${domain} is not yet supported for direct transcription. ` +
        "Try downloading the video first, then upload it via the Upload Files tab."
      );
    }

    // ── Direct file URLs: fetch + Groq ──
    if (!env.GROQ_API_KEY) {
      return jsonResponse({ error: "GROQ_API_KEY not configured on worker" }, 500);
    }

    const fileData = await downloadUrl(sourceUrl);

    if (fileData.size > MAX_GROQ_SIZE) {
      return jsonResponse({
        error: `File is ${(fileData.size / 1048576).toFixed(1)} MB (Groq limit is 25 MB). Download the file and use the Upload tab — it auto-chunks large files.`,
        source_type: sourceType,
      }, 413);
    }

    const transcript = await transcribeWithGroq(
      fileData.blob, fileData.filename, model, language, env.GROQ_API_KEY
    );

    return jsonResponse({
      transcript: transcript.text,
      segments: transcript.segments || [],
      language: transcript.language || "auto",
      duration: transcript.duration || 0,
      source_type: sourceType,
    });
  } catch (err) {
    return jsonResponse({ error: err.message, source_type: sourceType }, 500);
  }
}

// ── URL classification ───────────────────────────────────────

function classifyUrl(url) {
  const lower = url.toLowerCase();

  if (lower.includes("fathom.video") || lower.includes("fathom.ai")) return "fathom";
  if (lower.includes("youtube.com") || lower.includes("youtu.be")) return "youtube";
  if (
    ["vimeo.com", "tiktok.com", "twitter.com", "x.com", "loom.com",
     "twitch.tv", "dailymotion.com", "soundcloud.com", "facebook.com",
     "instagram.com", "reddit.com"].some((d) => lower.includes(d))
  ) return "platform";

  const path = lower.split("?")[0];
  if ([".mp4",".mp3",".wav",".m4a",".webm",".ogg",".flac",".avi",".mkv",".mov",".aac",".opus",".wma"]
      .some((ext) => path.endsWith(ext))) return "direct";

  if (lower.includes("sharepoint.com") || lower.includes("onedrive")) return "sharepoint";

  return "unknown";
}

// ── YouTube transcript extraction ────────────────────────────

function extractVideoId(url) {
  const u = new URL(url);
  if (u.hostname.includes("youtu.be")) return u.pathname.slice(1).split("/")[0];
  if (u.searchParams.has("v")) return u.searchParams.get("v");
  if (u.pathname.includes("/embed/")) return u.pathname.split("/embed/")[1].split(/[/?]/)[0];
  if (u.pathname.includes("/shorts/")) return u.pathname.split("/shorts/")[1].split(/[/?]/)[0];
  return null;
}

async function getYouTubeTranscript(url, preferredLang) {
  const videoId = extractVideoId(url);
  if (!videoId) throw new Error("Could not extract YouTube video ID from this URL.");

  // Fetch the YouTube watch page to get caption track info
  const watchResp = await fetch(`https://www.youtube.com/watch?v=${videoId}`, {
    headers: {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
      "Accept-Language": "en-US,en;q=0.9",
    },
  });

  if (!watchResp.ok) throw new Error(`YouTube returned HTTP ${watchResp.status}`);

  const html = await watchResp.text();

  // Extract video title
  let title = "YouTube Video";
  const titleMatch = html.match(/"title":"(.*?)"/);
  if (titleMatch) title = JSON.parse(`"${titleMatch[1]}"`);

  // Find captions data in the page
  const captionsMatch = html.match(/"captions":\s*(\{.*?"playerCaptionsTracklistRenderer".*?\})\s*,\s*"videoDetails"/s);
  if (!captionsMatch) {
    // Try alternate pattern
    const altMatch = html.match(/"captionTracks":\s*(\[.*?\])/s);
    if (!altMatch) {
      throw new Error(
        "No captions available for this YouTube video. " +
        "The video may not have subtitles/CC enabled. " +
        "Try downloading the video and uploading it via the Upload Files tab for AI transcription."
      );
    }
    const tracks = JSON.parse(altMatch[1]);
    return await fetchCaptionTrack(tracks, preferredLang, title);
  }

  // Parse the captions object to find caption tracks
  let captionsData;
  try {
    // Extract just the captionTracks array
    const tracksMatch = captionsMatch[1].match(/"captionTracks":\s*(\[.*?\])/s);
    if (!tracksMatch) throw new Error("No caption tracks found");
    captionsData = JSON.parse(tracksMatch[1]);
  } catch {
    throw new Error(
      "Could not parse YouTube captions data. " +
      "Try downloading the video and uploading it via the Upload Files tab."
    );
  }

  return await fetchCaptionTrack(captionsData, preferredLang, title);
}

async function fetchCaptionTrack(tracks, preferredLang, title) {
  if (!tracks || tracks.length === 0) {
    throw new Error("No caption tracks available for this video.");
  }

  // Pick the best track: prefer manual captions in requested language, then auto-generated
  let track = null;

  if (preferredLang) {
    // Exact language match (manual first)
    track = tracks.find((t) => t.languageCode === preferredLang && t.kind !== "asr");
    if (!track) track = tracks.find((t) => t.languageCode === preferredLang);
  }

  if (!track) {
    // English manual, then English auto, then first manual, then first auto
    track =
      tracks.find((t) => t.languageCode === "en" && t.kind !== "asr") ||
      tracks.find((t) => t.languageCode === "en") ||
      tracks.find((t) => t.kind !== "asr") ||
      tracks[0];
  }

  // Fetch the caption content as JSON3 format
  let captionUrl = track.baseUrl;
  if (!captionUrl.includes("fmt=")) captionUrl += "&fmt=json3";
  else captionUrl = captionUrl.replace(/fmt=\w+/, "fmt=json3");

  const captionResp = await fetch(captionUrl, {
    headers: {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    },
  });

  if (!captionResp.ok) {
    throw new Error("Failed to fetch YouTube captions. Try again or upload the video directly.");
  }

  const captionData = await captionResp.json();

  // Parse JSON3 format into segments
  const segments = [];
  const textParts = [];
  const events = captionData.events || [];

  for (const event of events) {
    if (!event.segs) continue;

    const text = event.segs.map((s) => s.utf8 || "").join("").trim();
    if (!text || text === "\n") continue;

    const startSec = (event.tStartMs || 0) / 1000;
    const endSec = startSec + (event.dDurationMs || 3000) / 1000;

    segments.push({
      start: Math.round(startSec * 100) / 100,
      end: Math.round(endSec * 100) / 100,
      text: text,
    });
    textParts.push(text);
  }

  if (segments.length === 0) {
    throw new Error("YouTube captions were empty. Try uploading the video directly.");
  }

  const duration = segments[segments.length - 1].end;

  return {
    transcript: textParts.join(" "),
    segments,
    language: track.languageCode || "en",
    duration: Math.round(duration * 100) / 100,
    title,
  };
}

// ── Fathom API ───────────────────────────────────────────────

async function tryFathomApi(url, apiKey) {
  const recordingId = url.replace(/\/+$/, "").split("/").pop();

  try {
    const resp = await fetch("https://api.fathom.ai/external/v1/meetings", {
      headers: { "X-Api-Key": apiKey },
    });

    if (!resp.ok) return null;

    const data = await resp.json();
    const items = data.items || data.meetings || [];

    for (const meeting of items) {
      const matchFields = [
        meeting.url, meeting.share_url, meeting.id,
        meeting.short_url, meeting.recording_url,
      ].filter(Boolean).map((s) => s.toLowerCase());

      if (matchFields.some((f) => f.includes(recordingId.toLowerCase()))) {
        const parts = [];
        for (const seg of meeting.transcript || []) {
          const speaker = seg.speaker?.display_name || seg.speaker_name || "Speaker";
          const text = seg.text || "";
          const ts = seg.timestamp || seg.start_time || "";
          parts.push(`[${ts}] ${speaker}: ${text}`);
        }

        return {
          transcript: parts.join("\n") || meeting.summary || "No transcript content",
          title: meeting.title || "Fathom Meeting",
        };
      }
    }
  } catch (err) {
    console.error("Fathom API error:", err.message);
  }

  return null;
}

// ── Download file from URL ───────────────────────────────────

async function downloadUrl(url) {
  const resp = await fetch(url, {
    headers: {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    },
    redirect: "follow",
  });

  if (!resp.ok) {
    throw new Error(`Failed to download: HTTP ${resp.status} from ${new URL(url).hostname}`);
  }

  const contentType = resp.headers.get("content-type") || "";
  const blob = await resp.blob();

  if (contentType.includes("text/html")) {
    throw new Error(
      "This URL returned a web page, not an audio/video file. " +
      "Paste a direct link to a media file, or use the Upload Files tab."
    );
  }

  const urlPath = url.split("?")[0].split("#")[0];
  const urlFilename = urlPath.split("/").pop() || "audio";
  const ext = urlFilename.includes(".")
    ? urlFilename.split(".").pop()
    : guessExtension(contentType);

  return { blob, filename: `download.${ext}`, size: blob.size };
}

function guessExtension(contentType) {
  if (contentType.includes("mp3") || contentType.includes("mpeg")) return "mp3";
  if (contentType.includes("mp4")) return "mp4";
  if (contentType.includes("wav")) return "wav";
  if (contentType.includes("ogg")) return "ogg";
  if (contentType.includes("webm")) return "webm";
  if (contentType.includes("flac")) return "flac";
  if (contentType.includes("m4a") || contentType.includes("mp4a")) return "m4a";
  return "mp4";
}

// ── Groq Whisper API ─────────────────────────────────────────

async function transcribeWithGroq(blob, filename, model, language, apiKey) {
  const formData = new FormData();
  formData.append("file", blob, filename);
  formData.append("model", model);
  formData.append("response_format", "verbose_json");
  formData.append("temperature", "0");
  if (language) formData.append("language", language);

  const resp = await fetch(GROQ_ENDPOINT, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}` },
    body: formData,
  });

  if (!resp.ok) {
    const errText = await resp.text();
    let msg = `Groq API error (HTTP ${resp.status})`;
    try { msg = JSON.parse(errText).error?.message || msg; } catch {}
    throw new Error(msg);
  }

  return resp.json();
}

// ── Helpers ──────────────────────────────────────────────────

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...CORS_HEADERS },
  });
}
