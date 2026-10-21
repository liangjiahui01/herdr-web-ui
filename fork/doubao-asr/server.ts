/**
 * fork: an OpenAI-shaped speech-to-text endpoint backed by Volcengine's Doubao ASR (bigmodel).
 *
 * The web UI's voice input posts the browser's clip to `<base_url>/audio/transcriptions` with the
 * key from its settings as a Bearer token (server/voice.ts). Point HERDR_WEB_OPENAI_BASE_URL at
 * this server and save the Volcengine key in Settings → Voice: the key comes with each request and
 * is passed on to Volcengine only, never stored or logged here.
 *
 * The key is either a new-console API key (sent as X-Api-Key) or `APP_ID:ACCESS_TOKEN` of the
 * old console (X-Api-App-Key / X-Api-Access-Key). The clip is decoded with ffmpeg to 16 kHz mono
 * PCM and sent over the streaming ASR WebSocket in 100 ms packets; the final text is answered as
 * JSON, which server/voice.ts accepts in place of a stream.
 *
 * Polish (`/chat/completions`) is not offered: Doubao ASR already drops filler words (enable_ddc),
 * and the web UI keeps the raw transcript when polish fails.
 */

import { randomUUID } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { gunzipSync } from "node:zlib";

const PORT = Number(process.env["DOUBAO_ASR_PORT"] ?? 7318);
const ENDPOINT = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async";
const MODEL_2 = "volc.seedasr.sauc.duration";
const MODEL_1 = "volc.bigasr.sauc.duration";
const RATE = 16000;
const PACKET = RATE * 2 / 10; // 100 ms of 16-bit mono
const MAX_AUDIO_BYTES = 30 * 1024 * 1024;
const HOTWORDS_MAX = 100;
const TIMEOUT_MS = 60_000;

class AsrError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json; charset=utf-8" } });
}

const failure = (status: number, message: string) => json({ error: { message, type: "doubao_asr" } }, status);

/** the Volcengine headers for a Bearer value: `APP_ID:ACCESS_TOKEN`, else an API key */
function authHeaders(bearer: string): Record<string, string> {
  const colon = bearer.indexOf(":");
  if (colon > 0) return { "X-Api-App-Key": bearer.slice(0, colon), "X-Api-Access-Key": bearer.slice(colon + 1) };
  return { "X-Api-Key": bearer };
}

/** any browser container (webm/opus, mp4/aac, ogg, wav…) to raw 16 kHz mono s16le */
async function toPcm(audio: Blob): Promise<Uint8Array> {
  // a file, not a pipe: an mp4 from Safari has its index at the end and needs seeking
  const dir = await mkdtemp(join(tmpdir(), "doubao-asr-"));
  try {
    const input = join(dir, "clip");
    await writeFile(input, new Uint8Array(await audio.arrayBuffer()));
    const proc = Bun.spawn(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", input, "-f", "s16le", "-ac", "1", "-ar", String(RATE), "pipe:1"], { stdout: "pipe", stderr: "pipe" });
    const [pcm, err, code] = await Promise.all([new Response(proc.stdout).arrayBuffer(), new Response(proc.stderr).text(), proc.exited]);
    if (code !== 0) throw new AsrError(400, `Could not decode the recording: ${err.trim().slice(0, 200)}`);
    return new Uint8Array(pcm);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

/** protocol v1, 4-byte header, then the payload's size and the payload */
function frame(typeAndFlags: number, serialization: number, payload: Uint8Array): Uint8Array {
  const out = new Uint8Array(8 + payload.length);
  out.set([0x11, typeAndFlags, serialization, 0]);
  new DataView(out.buffer).setUint32(4, payload.length);
  out.set(payload, 8);
  return out;
}

function requestPayload(hotwords: string[]): Uint8Array {
  const request: Record<string, unknown> = {
    model_name: "bigmodel",
    enable_punc: true,
    enable_ddc: true,
    enable_nonstream: true,
    result_type: "full",
  };
  if (hotwords.length > 0) request["corpus"] = { context: JSON.stringify({ hotwords: hotwords.map((word) => ({ word })) }) };
  const root = {
    user: { uid: "herdr-web-ui" },
    audio: { format: "pcm", codec: "raw", rate: RATE, bits: 16, channel: 1 },
    request,
  };
  return new TextEncoder().encode(JSON.stringify(root));
}

interface Reply { last: boolean; error: string | null; text: string | null }

function parse(data: Uint8Array): Reply | null {
  if (data.length < 8) return null;
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const headerSize = (data[0]! & 0x0f) * 4;
  const type = (data[1]! >> 4) & 0x0f;
  const flags = data[1]! & 0x0f;
  const compression = data[2]! & 0x0f;
  let at = headerSize;
  if (type === 0x0f) {
    const code = view.getUint32(at);
    const size = view.getUint32(at + 4);
    const message = new TextDecoder().decode(data.subarray(at + 8, at + 8 + size));
    return { last: true, error: `${code}${message ? `: ${message}` : ""}`, text: null };
  }
  if (flags === 0x01 || flags === 0x03) at += 4; // sequence number
  const size = view.getUint32(at);
  let payload = data.subarray(at + 4, at + 4 + size);
  if (compression === 1) payload = gunzipSync(payload);
  const body = JSON.parse(new TextDecoder().decode(payload)) as { result?: { text?: string; utterances?: { text?: string }[] }; text?: string };
  const text = body.result?.text || body.result?.utterances?.map((u) => u.text ?? "").join("") || body.text || "";
  return { last: flags === 0x02 || flags === 0x03, error: null, text };
}

/** one recognition over the WebSocket; rejects with `handshake` when the server refused the connection */
function recognize(pcm: Uint8Array, bearer: string, model: string, hotwords: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    let text = "";
    let opened = false;
    let settled = false;
    const done = (error: AsrError | null, value = "") => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try { ws.close(); } catch { /* already closed */ }
      error ? reject(error) : resolve(value);
    };
    const timer = setTimeout(() => done(new AsrError(504, "Doubao ASR did not answer in time")), TIMEOUT_MS);
    // Bun's WebSocket takes request headers
    const ws = new WebSocket(ENDPOINT, { headers: { ...authHeaders(bearer), "X-Api-Resource-Id": model, "X-Api-Connect-Id": randomUUID() } } as unknown as string[]);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      opened = true;
      ws.send(frame(0x10, 0x10, requestPayload(hotwords)));
      for (let at = 0; at < pcm.length; at += PACKET) {
        const end = Math.min(at + PACKET, pcm.length);
        ws.send(frame(end >= pcm.length ? 0x22 : 0x20, 0x00, pcm.subarray(at, end)));
      }
      if (pcm.length === 0) ws.send(frame(0x22, 0x00, new Uint8Array()));
    };
    ws.onmessage = (event) => {
      let reply: Reply | null;
      try { reply = parse(new Uint8Array(event.data as ArrayBuffer)); } catch { return; }
      if (!reply) return;
      if (reply.error !== null) {
        // after the audio, an error frame can be how the server ends the session
        if (text) done(null, text);
        else done(new AsrError(502, `Doubao ASR: ${reply.error}`));
        return;
      }
      if (reply.text !== null && (reply.text || reply.last)) text = reply.text;
      if (reply.last) done(null, text);
    };
    ws.onerror = () => {
      if (!opened) done(Object.assign(new AsrError(401, "Doubao ASR refused the connection (check the key, and that it has the streaming ASR model)"), { handshake: true }));
    };
    ws.onclose = () => done(text ? null : opened ? new AsrError(502, "Doubao ASR closed without a transcript") : new AsrError(401, "Doubao ASR refused the connection"), text);
  });
}

async function transcribe(request: Request): Promise<Response> {
  const bearer = request.headers.get("authorization")?.replace(/^Bearer\s+/i, "").trim();
  if (!bearer) return failure(401, "Missing the Volcengine key (Settings → Voice)");
  if (Number(request.headers.get("content-length") ?? 0) > MAX_AUDIO_BYTES) return failure(413, "The recording is too large");
  let form: FormData;
  try { form = await request.formData(); } catch { return failure(400, "Send the recording as multipart/form-data"); }
  const audio = form.get("file");
  if (audio === null || typeof audio === "string" || audio.size === 0) return failure(400, "The recording is empty");
  const asked = form.get("model");
  const model = typeof asked === "string" && asked.startsWith("volc.") ? asked : MODEL_2;
  const hotwords = [...new Set(form.getAll("keywords[]").filter((k): k is string => typeof k === "string").map((k) => k.trim()).filter((k) => k && k.length <= 50))].slice(0, HOTWORDS_MAX);
  try {
    const pcm = await toPcm(audio);
    let text: string;
    try {
      text = await recognize(pcm, bearer, model, hotwords);
    } catch (error) {
      // a key without access to model 2.0 tries 1.0 once, as the Android app does
      if (!(error as { handshake?: boolean }).handshake || model !== MODEL_2) throw error;
      text = await recognize(pcm, bearer, MODEL_1, hotwords);
    }
    return json({ text });
  } catch (error) {
    if (error instanceof AsrError) return failure(error.status, error.message);
    return failure(502, error instanceof Error ? error.message : String(error));
  }
}

Bun.serve({
  hostname: "127.0.0.1",
  port: PORT,
  async fetch(request) {
    const { pathname } = new URL(request.url);
    if (pathname === "/health") return json({ ok: true });
    if (request.method === "POST" && /\/audio\/transcriptions$/.test(pathname)) return transcribe(request);
    if (/\/chat\/completions$/.test(pathname)) return failure(501, "Polish is off: Doubao ASR already tidies the transcript");
    return failure(404, "not found");
  },
});
console.log(`doubao-asr listening on 127.0.0.1:${PORT}`);
