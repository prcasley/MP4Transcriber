/**
 * Web Worker for Whisper transcription using @huggingface/transformers.
 * Runs the Whisper model entirely in the browser via WebAssembly/WebGPU.
 */

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.1';

// Allow local model caching in browser
env.allowLocalModels = false;
env.useBrowserCache = true;

let transcriber = null;
let currentModelId = null;

const MODEL_MAP = {
    'tiny':   'onnx-community/whisper-tiny',
    'base':   'onnx-community/whisper-base',
    'small':  'onnx-community/whisper-small',
};

self.onmessage = async function(e) {
    const { type, audioData, modelSize, language, jobId } = e.data;

    if (type === 'transcribe') {
        try {
            const modelId = MODEL_MAP[modelSize] || MODEL_MAP['base'];

            // Load or switch model
            if (!transcriber || currentModelId !== modelId) {
                self.postMessage({ type: 'status', jobId, status: 'loading_model', progress: 0 });

                transcriber = await pipeline('automatic-speech-recognition', modelId, {
                    dtype: 'q8',
                    device: 'wasm',
                    progress_callback: (progress) => {
                        if (progress.status === 'progress' && progress.progress) {
                            self.postMessage({
                                type: 'status', jobId,
                                status: 'loading_model',
                                progress: Math.round(progress.progress),
                                detail: `Downloading model: ${Math.round(progress.progress)}%`
                            });
                        }
                    }
                });
                currentModelId = modelId;
            }

            self.postMessage({ type: 'status', jobId, status: 'transcribing', progress: 0 });

            // Transcribe with timestamps
            const result = await transcriber(audioData, {
                language: language || null,
                task: 'transcribe',
                return_timestamps: true,
                chunk_length_s: 30,
                stride_length_s: 5,
                // Progress callback for transcription
                callback_function: (output) => {
                    if (output && output.text) {
                        self.postMessage({
                            type: 'partial',
                            jobId,
                            text: output.text,
                        });
                    }
                },
            });

            // Build segments from chunks
            const segments = [];
            if (result.chunks && result.chunks.length > 0) {
                result.chunks.forEach((chunk, i) => {
                    const text = (chunk.text || '').trim();
                    if (!text) return;
                    segments.push({
                        index: i + 1,
                        start: chunk.timestamp?.[0] ?? 0,
                        end: chunk.timestamp?.[1] ?? 0,
                        text,
                    });
                });
            }

            // Full text
            const fullText = result.text || segments.map(s => s.text).join(' ');

            // Build SRT
            const srt = segments.map((s, i) => {
                const fmtTime = (sec) => {
                    const h = Math.floor(sec / 3600);
                    const m = Math.floor((sec % 3600) / 60);
                    const secs = Math.floor(sec % 60);
                    const ms = Math.round((sec % 1) * 1000);
                    return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(secs).padStart(2,'0')},${String(ms).padStart(3,'0')}`;
                };
                return `${i+1}\n${fmtTime(s.start)} --> ${fmtTime(s.end)}\n${s.text}\n`;
            }).join('\n');

            self.postMessage({
                type: 'completed',
                jobId,
                result: {
                    text: fullText,
                    srt,
                    json: { segments },
                    segment_count: segments.length,
                    language: language || 'auto',
                },
            });

        } catch (err) {
            self.postMessage({
                type: 'error',
                jobId,
                error: err.message || String(err),
            });
        }
    }
};
