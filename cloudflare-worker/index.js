/**
 * TranscribeHQ — Cloudflare Worker
 *
 * Accepts any URL, fetches the audio server-side (no CORS issues),
 * sends it to Groq Whisper API, and returns the transcript.
 *
 * Secrets (set via `npx wrangler secret put`):
 *   GROQ_API_KEY   — required
 *   FATHOM_API_KEY  — optional, for direct Fathom transcript retrieval
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
    // Handle CORS preflight
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

  if (!env.GROQ_API_KEY) {
    return jsonResponse({ error: "GROQ_API_KEY not configured on worker" }, 500);
  }

  const model = body.model || "whisper-large-v3-turbo";
  const language = body.language || undefined;
  const sourceType = classifyUrl(sourceUrl);

  try {
    // ── Fathom: try API first ──
    if (sourceType === "fathom" && env.FATHOM_API_KEY) {
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

    // ── Download audio from URL ──
    const fileData = await downloadUrl(sourceUrl);

    if (fileData.size > MAX_GROQ_SIZE) {
      return jsonResponse({
        error: `File too large (${(fileData.size / 1048576).toFixed(1)} MB). Groq limit is 25 MB. Download the file and use the Upload tab to auto-chunk.`,
        source_type: sourceType,
      }, 413);
    }

    // ── Send to Groq ──
    const transcript = await transcribeWithGroq(
      fileData.blob,
      fileData.filename,
      model,
      language,
      env.GROQ_API_KEY
    );

    return jsonResponse({
      transcript: transcript.text,
      segments: transcript.segments || [],
      language: transcript.language || "auto",
      duration: transcript.duration || 0,
      source_type: sourceType,
    });
  } catch (err) {
    return jsonResponse({
      error: err.message,
      source_type: sourceType,
    }, 500);
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
        // Found — build transcript with speaker names
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

  // Reject HTML pages (user pasted a webpage, not a file)
  if (contentType.includes("text/html")) {
    throw new Error(
      "This URL returned a web page, not an audio/video file. " +
      "For YouTube, Fathom, and other platforms, paste a direct download link to the media file."
    );
  }

  // Determine filename from URL
  const urlPath = url.split("?")[0].split("#")[0];
  const urlFilename = urlPath.split("/").pop() || "audio";
  const ext = urlFilename.includes(".")
    ? urlFilename.split(".").pop()
    : guessExtension(contentType);
  const filename = `download.${ext}`;

  return { blob, filename, size: blob.size };
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
    try {
      msg = JSON.parse(errText).error?.message || msg;
    } catch {}
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
