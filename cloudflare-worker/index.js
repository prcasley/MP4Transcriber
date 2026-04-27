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
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
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
    if (url.pathname.startsWith("/transcribe-status/") && request.method === "GET") {
      const jobId = url.pathname.slice("/transcribe-status/".length);
      return handleStatus(jobId, env);
    }
    if (url.pathname === "/yt-transcript" && request.method === "GET") return handleYtTranscript(url, env);
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

    // ── YouTube: try transcript from edge first (bypasses IP block) ──
    if (isYouTubeUrl(sourceUrl)) {
      try {
        const videoId = extractYouTubeId(sourceUrl);
        if (videoId) {
          const transcript = await fetchYouTubeTranscript(videoId);
          return jsonResponse({
            transcript: transcript.text,
            srt: transcript.srt,
            segments: [],
            language: transcript.language,
            duration: 0,
            source_type: "youtube",
            note: `Transcript fetched from YouTube captions (${transcript.segment_count} segments)`,
          });
        }
      } catch (err) {
        console.log(`[yt-transcript] Edge fetch failed: ${err.message}, falling back to Render`);
        // Fall through to Render server
      }
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

      // The Render server returns a job_id — return it so the browser can poll
      // /transcribe-status/:id directly. (Avoids exhausting Worker subrequest budget.)
      if (renderData.job_id) {
        return jsonResponse({ job_id: renderData.job_id, source_type: sourceType, polling: true });
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

// ── Status proxy (browser polls this; we proxy to Render) ───

async function handleStatus(jobId, env) {
  if (!jobId) return jsonResponse({ error: "Missing job_id" }, 400);
  if (!env.RENDER_URL) return jsonResponse({ error: "Transcription server is not configured." }, 500);
  try {
    const resp = await fetch(`${env.RENDER_URL}/api/status/${encodeURIComponent(jobId)}`);
    const data = await resp.json().catch(() => ({ error: `Status check failed: HTTP ${resp.status}` }));
    return jsonResponse(data, resp.status);
  } catch (err) {
    return jsonResponse({ error: `Status check failed: ${err.message}` }, 502);
  }
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

// ── YouTube transcript (fetched from Cloudflare edge) ───────

function isYouTubeUrl(url) {
  const lower = url.toLowerCase();
  return lower.includes("youtube.com") || lower.includes("youtu.be");
}

function extractYouTubeId(url) {
  const m = url.match(/(?:v=|youtu\.be\/)([A-Za-z0-9_-]{11})/);
  return m ? m[1] : null;
}

async function handleYtTranscript(url, env) {
  // API key auth
  if (env.TRANSCRIBE_API_KEY) {
    const key = url.searchParams.get("api_key") || "";
    if (key !== env.TRANSCRIBE_API_KEY) return jsonResponse({ error: "Unauthorized" }, 401);
  }
  const videoId = url.searchParams.get("v");
  if (!videoId) return jsonResponse({ error: "Missing ?v=VIDEO_ID" }, 400);
  try {
    const transcript = await fetchYouTubeTranscript(videoId);
    return jsonResponse(transcript);
  } catch (err) {
    return jsonResponse({ error: err.message }, 500);
  }
}

async function fetchYouTubeTranscript(videoId) {
  const INNERTUBE_CONTEXT = {
    client: { clientName: "WEB", clientVersion: "2.20241126.01.00", hl: "en", gl: "US" },
  };

  // Try multiple caption languages via YouTube's timedtext API
  const langs = ["en", "en-US", "a.en"]; // a.en = auto-generated English
  let xml = null;
  let language = "en";

  for (const lang of langs) {
    const url = `https://www.youtube.com/api/timedtext?v=${videoId}&lang=${lang}&fmt=srv3`;
    const resp = await fetch(url, {
      headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36" },
    });
    if (resp.ok) {
      const body = await resp.text();
      if (body && body.includes("<text")) { xml = body; language = lang.replace("a.", ""); break; }
    }
  }

  // If simple langs failed, try Innertube to discover available tracks
  if (!xml) {
    const playerResp = await fetch("https://www.youtube.com/youtubei/v1/player?prettyPrint=false", {
      method: "POST",
      headers: { "Content-Type": "application/json", "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36" },
      body: JSON.stringify({ context: { client: { clientName: "WEB", clientVersion: "2.20241126.01.00", hl: "en", gl: "US" } }, videoId }),
    });
    if (playerResp.ok) {
      const pd = await playerResp.json();
      const tracks = pd?.captions?.playerCaptionsTracklistRenderer?.captionTracks;
      if (tracks && tracks.length > 0) {
        const track = tracks.find((t) => t.languageCode === "en") || tracks[0];
        const captResp = await fetch(track.baseUrl + "&fmt=srv3", {
          headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" },
        });
        if (captResp.ok) { xml = await captResp.text(); language = track.languageCode; }
      }
    }
  }

  if (!xml) throw new Error("No captions available for this video");
  // Parse XML — <text start="0.0" dur="1.5">Hello</text>
  const segments = [];
  const re = /<text\s+start="([^"]+)"\s+dur="([^"]+)"[^>]*>([\s\S]*?)<\/text>/g;
  let m;
  while ((m = re.exec(xml)) !== null) {
    const start = parseFloat(m[1]);
    const duration = parseFloat(m[2]);
    const text = m[3]
      .replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
      .replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&nbsp;/g, " ")
      .replace(/<[^>]+>/g, "").replace(/\n/g, " ").trim();
    if (text) segments.push({ start, duration, text });
  }
  if (segments.length === 0) throw new Error("No transcript segments found in caption data");

  const fullText = segments.map((s) => s.text).join(" ");
  const srt = segments.map((s, i) => {
    return `${i + 1}\n${fmtSrt(s.start)} --> ${fmtSrt(s.start + s.duration)}\n${s.text}\n`;
  }).join("\n");

  return { text: fullText, srt, language, segment_count: segments.length };
}

function fmtSrt(sec) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 1000);
  return `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")},${String(ms).padStart(3,"0")}`;
}
