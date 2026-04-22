/**
 * YouTube Transcript Proxy — Google Apps Script
 *
 * Deploy as web app: Deploy → New deployment → Web app → Anyone
 * Copy the URL and set it as TRANSCRIPT_PROXY_URL in Render env vars.
 */

function doGet(e) {
  var videoId = e.parameter.v;
  var apiKey = e.parameter.key || "";

  // Optional: protect with a simple key
  var SECRET = PropertiesService.getScriptProperties().getProperty("API_KEY") || "";
  if (SECRET && apiKey !== SECRET) {
    return ContentService.createTextOutput(JSON.stringify({ error: "Unauthorized" }))
      .setMimeType(ContentService.MimeType.JSON);
  }

  if (!videoId) {
    return ContentService.createTextOutput(JSON.stringify({ error: "Missing ?v=VIDEO_ID" }))
      .setMimeType(ContentService.MimeType.JSON);
  }

  try {
    var result = fetchTranscript(videoId);
    return ContentService.createTextOutput(JSON.stringify(result))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return ContentService.createTextOutput(JSON.stringify({ error: err.message }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

function fetchTranscript(videoId) {
  // Step 1: Fetch the YouTube page
  var html = UrlFetchApp.fetch("https://www.youtube.com/watch?v=" + videoId, {
    headers: {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
      "Accept-Language": "en-US,en;q=0.9",
    },
    muteHttpExceptions: true,
  }).getContentText();

  // Step 2: Extract ytInitialPlayerResponse
  var match = html.match(/var\s+ytInitialPlayerResponse\s*=\s*(\{.+?\})\s*;/s);
  if (!match) throw new Error("Could not find player response in YouTube page");

  var player = JSON.parse(match[1]);
  var tracks = (player.captions || {}).playerCaptionsTracklistRenderer || {};
  var captionTracks = tracks.captionTracks || [];

  if (captionTracks.length === 0) throw new Error("No captions available for this video");

  // Step 3: Pick English or first available
  var track = null;
  for (var i = 0; i < captionTracks.length; i++) {
    if (captionTracks[i].languageCode === "en") { track = captionTracks[i]; break; }
  }
  if (!track) {
    for (var i = 0; i < captionTracks.length; i++) {
      if (captionTracks[i].languageCode && captionTracks[i].languageCode.indexOf("en") === 0) { track = captionTracks[i]; break; }
    }
  }
  if (!track) track = captionTracks[0];

  // Step 4: Fetch caption XML
  var captUrl = track.baseUrl + "&fmt=srv3";
  var xml = UrlFetchApp.fetch(captUrl, { muteHttpExceptions: true }).getContentText();

  // Step 5: Parse XML
  var segments = [];
  var re = /<text\s+start="([^"]+)"\s+dur="([^"]+)"[^>]*>([\s\S]*?)<\/text>/g;
  var m;
  while ((m = re.exec(xml)) !== null) {
    var start = parseFloat(m[1]);
    var duration = parseFloat(m[2]);
    var text = m[3]
      .replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
      .replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&nbsp;/g, " ")
      .replace(/<[^>]+>/g, "").replace(/\n/g, " ").trim();
    if (text) segments.push({ start: start, duration: duration, text: text });
  }

  if (segments.length === 0) throw new Error("No transcript segments found");

  // Build output
  var fullText = segments.map(function(s) { return s.text; }).join(" ");
  var srtParts = segments.map(function(s, i) {
    return (i + 1) + "\n" + fmtSrt(s.start) + " --> " + fmtSrt(s.start + s.duration) + "\n" + s.text + "\n";
  });

  return {
    text: fullText,
    srt: srtParts.join("\n"),
    language: track.languageCode,
    segment_count: segments.length,
  };
}

function fmtSrt(sec) {
  var h = Math.floor(sec / 3600);
  var m = Math.floor((sec % 3600) / 60);
  var s = Math.floor(sec % 60);
  var ms = Math.floor((sec % 1) * 1000);
  return pad(h) + ":" + pad(m) + ":" + pad(s) + "," + pad3(ms);
}

function pad(n) { return n < 10 ? "0" + n : "" + n; }
function pad3(n) { return n < 10 ? "00" + n : n < 100 ? "0" + n : "" + n; }

// Quick test from the script editor
function testTranscript() {
  var result = fetchTranscript("fJ9rUzIMcZQ");
  Logger.log("Segments: " + result.segment_count);
  Logger.log("First 200 chars: " + result.text.substring(0, 200));
}
