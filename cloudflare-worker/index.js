/**
 * TranscribeHQ — Cloudflare Worker (Router)
 *
 * Routes URL transcription requests:
 *   - YouTube / video platforms → Render server (yt-dlp + Groq)
 *   - Fathom → Fathom API (with speaker names)
 *   - Direct files → fetches audio, sends to Groq Whisper
 *
 * Secrets:
 *   GROQ_API_KEY     — required for direct file transcription
 *   FATHOM_API_KEY   — optional, for Fathom meeting transcripts
 *   RENDER_URL       — required, URL of the Render server (e.g. https://transcribehq-server.onrender.com)
 */

const GROQ_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions";
const MAX_GROQ_SIZE = 25 * 1024 * 1024;

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, X-API-Key",
};

// Platforms that need yt-dlp (routed to Render server)
const YTDLP_PLATFORMS = [
  "youtube.com", "youtu.be", "vimeo.com", "tiktok.com", "twitter.com",
  "x.com", "loom.com", "twitch.tv", "dailymotion.com", "soundcloud.com",
  "facebook.com", "instagram.com", "reddit.com", "rumble.com", "bitchute.com",
  "odysee.com", "bandcamp.com", "spotify.com",
];

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return new Response(null, { headers: CORS_HEADERS });
    const url = new URL(request.url);
    if (url.pathname === "/transcribe-url" && request.method === "POST") return handleTranscribe(request, env);
    if (url.pathname === "/health") return jsonResponse({
      status: "ok",
      groq_key_set: !!env.GROQ_API_KEY,
      fathom_key_set: !!env.FATHOM_API_KEY,
      render_url_set: !!env.RENDER_URL,
      auth_required: !!env.TRANSCRIBE_API_KEY,
    });
    return jsonResponse({ error: "Not found. POST to /transcribe-url" }, 404);
  },
};

// ── Main handler ─────────────────────────────────────────────

async function handleTranscribe(request, env) {
  // API key auth (skipped if TRANSCRIBE_API_KEY not set)
  if (env.TRANSCRIBE_API_KEY) {
    const key = request.headers.get("X-API-Key") || new URL(request.url).searchParams.get("api_key") || "";
    if (key !== env.TRANSCRIBE_API_KEY) return jsonResponse({ error: "Unauthorized — invalid or missing API key" }, 401);
  }

  let body;
  try { body = await request.json(); } catch { return jsonResponse({ error: "Invalid JSON body" }, 400); }

  const sourceUrl = (body.url || "").trim();
  if (!sourceUrl) return jsonResponse({ error: "No URL provided" }, 400);

  const model = body.model || "whisper-large-v3-turbo";
  const language = body.language || undefined;
  const sourceType = classifyUrl(sourceUrl);

  try {
    // ── Fathom: use API directly ──
    if (sourceType === "fathom") {
      if (env.FATHOM_API_KEY) {
        const r = await tryFathomApi(sourceUrl, env.FATHOM_API_KEY);
        if (r) return jsonResponse({ transcript: r.transcript, source_type: "fathom", title: r.title, note: "Transcript from Fathom API (includes speaker names)" });
      }
      // Fall through to Render server if no API key
    }

    // ── YouTube / video platforms / SharePoint: route to Render server ──
    if (sourceType === "platform" || sourceType === "fathom" || sourceType === "sharepoint") {
      if (!env.RENDER_URL) {
        throw new Error("Video platform transcription server is not configured. Contact the site admin.");
      }

      const renderHeaders = { "Content-Type": "application/json" };
      // RENDER_API_KEY is the key for the Render server (separate from TRANSCRIBE_API_KEY which guards this Worker)
      const renderKey = env.RENDER_API_KEY || env.TRANSCRIBE_API_KEY;
      if (renderKey) renderHeaders["X-API-Key"] = renderKey;

      const renderResp = await fetch(`${env.RENDER_URL}/api/transcribe-url`, {
        method: "POST",
        headers: renderHeaders,
        body: JSON.stringify({ url: sourceUrl, model, language }),
      });

      if (!renderResp.ok) {
        const err = await renderResp.json().catch(() => ({ error: `Server error: ${renderResp.status}` }));
        throw new Error(err.error || `Transcription server error: ${renderResp.status}`);
      }

      const renderData = await renderResp.json();

      // The Render server returns a job_id — we need to poll for completion
      if (renderData.job_id) {
        return await pollRenderJob(env.RENDER_URL, renderData.job_id, sourceType);
      }

      // Or it might return the transcript directly
      return jsonResponse({ ...renderData, source_type: sourceType });
    }

    // ── Direct file URLs: fetch + Groq (handled in Worker) ──
    if (!env.GROQ_API_KEY) return jsonResponse({ error: "GROQ_API_KEY not configured" }, 500);

    const fileData = await downloadUrl(sourceUrl);
    if (fileData.size > MAX_GROQ_SIZE) {
      return jsonResponse({
        error: `File is ${(fileData.size / 1048576).toFixed(1)} MB (limit 25 MB). Download and use Upload tab — it auto-chunks.`,
        source_type: sourceType,
      }, 413);
    }

    const transcript = await transcribeWithGroq(fileData.blob, fileData.filename, model, language, env.GROQ_API_KEY);
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

// ── Poll Render server for job completion ────────────────────

async function pollRenderJob(renderUrl, jobId, sourceType) {
  const maxWait = 300000; // 5 minutes max (SharePoint downloads + large files need time)
  const start = Date.now();

  while (Date.now() - start < maxWait) {
    const resp = await fetch(`${renderUrl}/api/status/${jobId}`);
    if (!resp.ok) throw new Error("Failed to check transcription status.");

    const job = await resp.json();

    if (job.status === "completed" && job.result) {
      return jsonResponse({
        transcript: job.result.text || "",
        segments: job.result.json?.segments || [],
        language: job.result.language || "auto",
        duration: job.result.duration || 0,
        source_type: sourceType,
        note: `Transcribed via Groq Whisper (${job.result.chunks_processed || 1} chunk(s))`,
      });
    }

    if (job.status === "error") {
      throw new Error(job.error || "Transcription failed on server.");
    }

    // Wait 2 seconds before polling again
    await new Promise((r) => setTimeout(r, 2000));
  }

  throw new Error("Transcription timed out. The file may be too large. Try a shorter video.");
}

// ── URL classification ───────────────────────────────────────

function classifyUrl(url) {
  const lower = url.toLowerCase();
  if (lower.includes("fathom.video") || lower.includes("fathom.ai")) return "fathom";
  if (YTDLP_PLATFORMS.some((d) => lower.includes(d))) return "platform";
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
    const resp = await fetch("https://api.fathom.ai/external/v1/meetings?include_transcript=true", { headers: { "X-Api-Key": apiKey } });
    if (!resp.ok) return null;
    const data = await resp.json();
    for (const meeting of (data.items || data.meetings || [])) {
      const fields = [meeting.url, meeting.share_url, meeting.id, meeting.short_url, meeting.recording_url].filter(Boolean).map((s) => s.toLowerCase());
      if (fields.some((f) => f.includes(recordingId.toLowerCase()))) {
        const parts = (meeting.transcript || []).map((seg) => {
          const speaker = seg.speaker?.display_name || seg.speaker_name || "Speaker";
          return `[${seg.timestamp || seg.start_time || ""}] ${speaker}: ${seg.text || ""}`;
        });
        return { transcript: parts.join("\n") || meeting.summary || "No transcript content", title: meeting.title || "Fathom Meeting" };
      }
    }
  } catch (err) { console.error("Fathom API error:", err.message); }
  return null;
}

// ── Download file from URL ───────────────────────────────────

async function downloadUrl(url) {
  const resp = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" }, redirect: "follow" });
  if (!resp.ok) throw new Error(`Failed to download: HTTP ${resp.status} from ${new URL(url).hostname}`);
  const contentType = resp.headers.get("content-type") || "";
  const blob = await resp.blob();
  if (contentType.includes("text/html")) throw new Error("This URL returned a web page, not an audio/video file. Paste a direct media link, or use the Upload Files tab.");
  const urlPath = url.split("?")[0].split("#")[0];
  const urlFilename = urlPath.split("/").pop() || "audio";
  const ext = urlFilename.includes(".") ? urlFilename.split(".").pop() : guessExt(contentType);
  return { blob, filename: `download.${ext}`, size: blob.size };
}

function guessExt(ct) {
  if (ct.includes("mp3") || ct.includes("mpeg")) return "mp3";
  if (ct.includes("mp4")) return "mp4";
  if (ct.includes("wav")) return "wav";
  if (ct.includes("ogg")) return "ogg";
  if (ct.includes("webm")) return "webm";
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
  const resp = await fetch(GROQ_ENDPOINT, { method: "POST", headers: { Authorization: `Bearer ${apiKey}` }, body: formData });
  if (!resp.ok) { const t = await resp.text(); let msg = `Groq error (${resp.status})`; try { msg = JSON.parse(t).error?.message || msg; } catch {} throw new Error(msg); }
  return resp.json();
}

// ── Helpers ──────────────────────────────────────────────────

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json", ...CORS_HEADERS } });
}
