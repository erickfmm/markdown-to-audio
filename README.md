# Markdown to Audio (md-tts)

Convert Markdown documents into WAV audio (English/Spanish) using local TTS engines.

Text is split by sections (`#` headings) and paragraphs; each paragraph is synthesized individually and fragments are joined with a configurable pause to produce one audio file per document.

The project ships as **three interchangeable frontends over one core service**:

- `md-tts` — batch **CLI** that converts every `.md` file in a directory.
- `md-tts-web` — **web GUI** (Flask) + **REST API backend** (FastAPI) with live progress, batch conversion and job history.
- `md_tts.service` — importable **Python library** (engine caching, progress callbacks).

```text
                        ┌────────────────────────────┐
   CLI (md-tts) ───────►│                            │
                        │  md_tts.service            │
   API (FastAPI) ──────►│  · Markdown parser         │────► TTS engines
                        │  · engine cache            │      (MMS, Kokoro,
   Library (import) ───►│  · synthesis + concat      │       Chatterbox,
                        └────────────────────────────┘       VibeVoice,
                                                             CosyVoice,
                                                             Qwen3-TTS)
        Browser ──► Flask GUI ──proxy /api/*──► FastAPI ──► JobManager (FIFO)
```

**Web GUI features:** drag & drop batch upload or pasted text · engine/language/pause/workers/device options · Qwen3 voice options (model size, premium speaker, style instruction, voice design description, voice cloning via file upload or browser recording) · live per-job progress bars · job history with inline audio player, download and delete · bilingual interface (ES/EN) · backend health indicator.

## Table of contents

- [Supported engines](#supported-engines)
- [Voice cloning and voice design (Qwen3-TTS)](#voice-cloning-and-voice-design-qwen3-tts)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [CLI usage (`md-tts`)](#cli-usage-md-tts)
- [Web GUI](#web-gui)
- [Ports and URLs](#ports-and-urls)
- [REST API reference](#rest-api-reference)
- [Docker](#docker)
- [`start.sh` reference](#startsh-reference)
- [Configuration reference](#configuration-reference)
- [Project structure](#project-structure)
- [Development notes](#development-notes)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Model licenses](#model-licenses)
- [Documentación en español](#markdown-a-audio-español)

## Supported engines

| Engine | Model | Languages | Notes |
| --- | --- | --- | --- |
| `mms` *(default)* | [facebook/mms-tts-spa](https://huggingface.co/facebook/mms-tts-spa) / [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng) | es, en | Lightest and most reliable option; CPU-friendly. |
| `kokoro` | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | es, en | Needs `espeak-ng` installed for phonemization. |
| `chatterbox` | [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) | es (multilingual build), en | Falls back to the English-only build if the multilingual package is unavailable. |
| `vibevoice` | [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) | en | Optimized for English; Spanish quality may degrade (a warning is shown). `--language` is ignored. |
| `cosyvoice` | [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | es, en | Requires installing [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) from its repo; supports a reference voice WAV. |
| `qwen3-customvoice` | [Qwen3-TTS-12Hz-1.7B/0.6B-CustomVoice](https://huggingface.co/collections/Qwen/qwen3-tts) | es, en *(+8 more)* | 9 premium voices + optional style instruction (1.7B only). Needs the optional `qwen` extra. |
| `qwen3-voicedesign` | [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | es, en *(+8 more)* | Designs a brand-new voice from a natural-language description (required). Needs the optional `qwen` extra. |
| `qwen3-clone` | [Qwen3-TTS-12Hz-1.7B/0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | es, en *(+8 more)* | 3-second voice cloning from a reference audio (WAV/MP3 upload or browser recording). Needs the optional `qwen` extra. |

Language behavior:

- `--language` is applied by `mms`, `kokoro`, `chatterbox` (via `language_id`), `cosyvoice` (language-specific prompt template) and the `qwen3-*` engines (`es`→Spanish, `en`→English; Qwen3-TTS itself supports 10 languages).
- `vibevoice` is English-first; using `--language es` logs a warning.
- Default language is `es`.

Device behavior:

- `--device cpu` forces CPU; `--device cuda` forces GPU (falls back to CPU with a warning if unavailable).
- `--device auto` (and the web UI's *auto*) picks CUDA when available, else CPU. For the `qwen3-*` engines, leaving `--device` unset also means *auto*.
- Qwen3 engines run in `bfloat16` on GPU and `float32` on CPU automatically.

### Qwen3-TTS notes

- **Install:** the engines need the optional dependency — `uv sync --no-default-groups --extra qwen` or `pip install 'md-tts[qwen]'` (package: [`qwen-tts`](https://github.com/QwenLM/Qwen3-TTS)). ⚠️ `qwen-tts` pins exact versions of `transformers` (4.57.3) and `accelerate` (1.12.0) and needs the `sox` system binary — that's why it's an opt-in extra instead of a base dependency, and why it can't coexist with `chatterbox` (see [Installation](#installation)). The Docker images accept `--build-arg TTS_PROFILE=qwen` to include it.
- **Model sizes:** `--qwen-model 1.7b` (default, best quality, ~4 GB VRAM in bf16) or `0.6b` (faster, ~2 GB). Any full HuggingFace id or local path also works (e.g. fine-tunes of the Base model).
- **Voices (CustomVoice):** Vivian, Serena, Uncle_Fu, Dylan, Eric (Chinese), Ryan, Aiden (English), Ono_Anna (Japanese), Sohee (Korean) — each can speak any supported language; native-language use is recommended.
- **Streaming:** this project synthesizes paragraph-by-paragraph (non-streaming API).
- **Tokenizer:** the `Qwen3-TTS-Tokenizer-12Hz` codec is downloaded automatically with the model; nothing to configure.

## Voice cloning and voice design (Qwen3-TTS)

### Cloning a voice (`qwen3-clone`)

The Base models clone a voice from a short reference clip plus its transcript.

**What makes a good reference clip:**

- 3–10 seconds, a **single speaker**, no music or background noise.
- Clean, natural read; WAV or MP3 (the backend normalizes everything to mono 24 kHz WAV; compressed formats need FFmpeg).
- A transcript of exactly what is said (**`--qwen-ref-text`** / `qwen_ref_text`) — cloning quality drops notably without it.

**CLI:**

```bash
md-tts --engine qwen3-clone --language es --device auto \
  --qwen-ref-audio mi_voz.wav \
  --qwen-ref-text "Hola, esta es mi voz de referencia." \
  --input-dir modificacion1 --output-dir output_audio
```

**Web GUI:** pick `qwen3-clone`, then upload an MP3/WAV **or record directly from the browser** (🎤 button — requires `localhost` or HTTPS), optionally type the transcript, preview the audio and convert. The clip is uploaded once to `/api/voice-references` and can be reused across jobs.

If you don't have a transcript, `--qwen-xvector-only` / `qwen_xvector_only=true` clones using only the speaker embedding (lower quality).

### Designing a voice (`qwen3-voicedesign`)

VoiceDesign creates a voice from scratch out of a **description** — no reference audio needed. The instruction (`--qwen-instruct` / `qwen_instruct`, required) works best when it covers:

1. **Gender and age** — "adult female voice", "male, 17 years old".
2. **Pitch/timbre** — "warm timbre", "bright, slightly edgy", "low, mellow", "tenor range".
3. **Emotion/tone** — "calm documentary tone", "incredulous with a hint of panic", "cheerful and energetic".
4. **Pace and delivery** — "slow, deliberate pace", "fast rhythmic drive".

Examples (from the model cards):

- 🇬🇧 *"Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."*
- 🇬🇧 *"Male, 17 years old, tenor range, gaining confidence — deeper breath support now, though vowels still tighten when nervous."*
- 🇪🇸 *"Voz femenina adulta, timbre cálido y grave, tono sereno de narradora de documental, ritmo pausado."*

```bash
md-tts --engine qwen3-voicedesign --language en --device cuda \
  --qwen-instruct "Young male voice, tenor range, warm timbre, calm documentary tone" \
  --input-dir modificacion1 --output-dir output_audio
```

### Styling a preset voice (`qwen3-customvoice`)

The 1.7B CustomVoice model also accepts an optional style instruction on top of the chosen speaker:

```bash
md-tts --engine qwen3-customvoice --qwen-speaker Ryan --language en \
  --qwen-instruct "Very happy." --input-dir modificacion1 --output-dir output_audio
```

(The 0.6B CustomVoice model ignores/rejects instructions — use `1.7b` for instruct support.)

## Requirements

| Requirement | Details |
| --- | --- |
| Python | 3.10 – 3.12 |
| FFmpeg | Required by `pydub` (audio concatenation/export, decoding uploaded MP3/WebM reference audio) |
| espeak-ng | Only needed for the `kokoro` engine |
| sox | Only needed for the `qwen3-*` engines (`qwen-tts` shell dependency) |
| Internet | First run per engine downloads model weights (can be GBs) |
| GPU (optional) | NVIDIA GPU + CUDA PyTorch for `--device cuda`; CPU works everywhere |
| Docker (optional) | For the containerized web stack; NVIDIA Container Toolkit for GPU |
| uv (optional) | Only needed for `./start.sh local` and `./start.sh hybrid` |
| `qwen-tts` (optional) | Only needed by the `qwen3-*` engines — install the `qwen` extra |

## Installation

```bash
git clone <repo-url> markdown-to-audio
cd markdown-to-audio
```

With **uv** (recommended — used by `start.sh`):

```bash
uv sync                                  # base + kokoro + chatterbox (as before)
uv sync --no-default-groups --extra qwen # base + Qwen3-TTS (no chatterbox)
```

With **pip**:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e .              # base engines
python3 -m pip install 'md-tts[qwen]'    # + Qwen3-TTS engines (replaces chatterbox)
```

> ⚠️ **`qwen3-*` and `chatterbox` cannot be installed together**: `qwen-tts` pins `transformers==4.57.3` while `chatterbox-tts` pins `transformers==4.46.3`/`5.2.0`. Engines unavailable in the current environment raise a clear `ImportError` at first use; the web UI still lists them but jobs will fail with that message. Choose the extra that matches your engines.

System dependencies on Debian/Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg espeak-ng
```

This installs two console scripts: `md-tts` (CLI) and `md-tts-web` (web server).

## Quick start

**CLI** — convert every `.md` in a folder:

```bash
uv run md-tts --input-dir modificacion1 --output-dir output_audio --engine mms
```

**Web GUI** — everything local in a single process (GUI on `:5000`, API on `:5001`):

```bash
uv sync
uv run md-tts-web --host 127.0.0.1 --port 5000
# open http://127.0.0.1:5000
```

**Web GUI** — full Docker stack (GUI on `:5000`, API on `:8000`):

```bash
./start.sh docker
```

## CLI usage (`md-tts`)

```bash
uv run md-tts --help
```

| Flag | Default | Description |
| --- | --- | --- |
| `--input-dir` | `<repo>/modificacion1` | Directory with `.md` files (processed in sorted order). |
| `--output-dir` | `<repo>/output_audio` | Where WAV files are written (`<name>.wav` per document). |
| `--engine` | `mms` | `mms` · `kokoro` · `chatterbox` · `vibevoice` · `cosyvoice` · `qwen3-customvoice` · `qwen3-voicedesign` · `qwen3-clone` |
| `--language` | `es` | `es` or `en` (engines that support it). |
| `--pause-ms` | `500` | Silence inserted between paragraphs, in milliseconds. |
| `--device` | auto | `cpu`, `cuda` or `auto` (GPU if available). |
| `--workers` | `1` | Parallel paragraph synthesis via multiprocessing (`>1` disables the engine cache: each worker loads its own engine). |
| `--save-fragments` | off | Keep one WAV per paragraph under `output_dir/fragments/<name>/` — **re-running resumes** from existing fragments. |
| `--cosyvoice-model-dir` | download | Local CosyVoice model directory (skips the HF download). |
| `--cosyvoice-prompt-wav` | — | Reference voice WAV for CosyVoice. |
| `--qwen-model` | `1.7b` | `qwen3-*`: model size `1.7b`/`0.6b`, or a full HuggingFace id / local path (fine-tunes). |
| `--qwen-speaker` | `Vivian` | `qwen3-customvoice`: premium voice (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee). |
| `--qwen-instruct` | — | `qwen3-customvoice`: style instruction (1.7b only). `qwen3-voicedesign`: voice description (**required**). |
| `--qwen-ref-audio` | — | `qwen3-clone`: reference audio (WAV/MP3, 3-10 s, single voice). |
| `--qwen-ref-text` | — | `qwen3-clone`: transcript of the reference audio (recommended). |
| `--qwen-xvector-only` | off | `qwen3-clone`: clone from the speaker embedding only (no transcript). |

Examples:

```bash
# Spanish with MMS (defaults)
md-tts --input-dir modificacion1 --output-dir output_audio

# English with MMS
md-tts --input-dir modificacion1 --output-dir output_audio_en --engine mms --language en

# Kokoro on GPU, saving fragments to allow resuming interrupted runs
md-tts --engine kokoro --device cuda --save-fragments --language es \
  --input-dir modificacion1 --output-dir output_audio

# Parallel paragraphs (one engine per worker)
md-tts --engine kokoro --language en --workers 4 --input-dir modificacion1 --output-dir out

# CosyVoice with local model + reference voice
md-tts --engine cosyvoice --language es --device cuda \
  --cosyvoice-model-dir /path/to/cosyvoice_model --cosyvoice-prompt-wav /path/to/voice.wav

# Qwen3: preset voice + style instruction (English, GPU auto-detected)
md-tts --engine qwen3-customvoice --qwen-speaker Ryan --language en \
  --qwen-instruct "Very happy." --input-dir modificacion1 --output-dir out

# Qwen3: design a voice from a text description
md-tts --engine qwen3-voicedesign --language en --device cuda \
  --qwen-instruct "Young male voice, tenor range, warm timbre, calm documentary tone" \
  --input-dir modificacion1 --output-dir out

# Qwen3: clone a voice from a 3-10 s reference clip (+ transcript)
md-tts --engine qwen3-clone --language es --device auto \
  --qwen-ref-audio mi_voz.wav --qwen-ref-text "Hola, esta es mi voz de referencia." \
  --input-dir modificacion1 --output-dir out

# How I use it
uv sync
uv run md-tts --input-dir modificacion1 --output-dir output_audio --engine kokoro \
  --device cuda --pause-ms 500 --save-fragments --workers 10 --language es
```

How parsing works: each `#`/`##`… heading is announced as its own short paragraph, text blocks are split on blank lines, and every paragraph becomes `NNNN_<section>.wav` when `--save-fragments` is used. Sections and fragment order are preserved; failures on individual paragraphs are logged and skipped (they don't abort the run).

## Web GUI

### Running it

| Mode | Command | What runs |
| --- | --- | --- |
| **local** (single process) | `uv run md-tts-web --host 127.0.0.1 --port 5000` | Flask GUI + embedded uvicorn/FastAPI thread (API at `127.0.0.1:<api-port>`). |
| **frontend-only** | `uv run md-tts-web --port 5000 --api-url http://127.0.0.1:8000` | Just the GUI proxying to an external backend (what the frontend container does). |
| **docker** | `./start.sh docker` | Both services as containers via `docker compose up --build`. |
| **hybrid** | `./start.sh hybrid` | Frontend in Docker, backend on the host via `uv run uvicorn …` (models/cache stay on the host). |

`md-tts-web` flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Bind address for the Flask GUI. Use `0.0.0.0` to expose on the LAN. |
| `--port` | `5000` | GUI port. |
| `--api-host` | `127.0.0.1` | Bind address of the embedded FastAPI backend. |
| `--api-port` | `--port + 1` | Port of the embedded backend. |
| `--api-url` | — | External backend URL; skips launching the embedded backend. |
| `--jobs-dir` | `output_audio/web` | Where the backend writes job audio (`<jobs-dir>/<job-id>/<name>.wav`). |

### Using the GUI

1. Open `http://localhost:5000` (ES/EN toggle sits in the top-right corner; the choice persists in the browser).
2. Drop/select one or **many** `.md`/`.markdown`/`.txt` files, **or** type a document name and paste Markdown text.
3. Pick options: engine, language, pause (ms), workers, device.
   - With a `qwen3-*` engine, extra options appear: **model size** (1.7B/0.6B), **premium speaker** (customvoice), **style instruction / voice description**, and for `qwen3-clone` a **reference audio** — upload an MP3/WAV or **record from the browser** (🎤), preview it and optionally type its transcript.
   - Browser recording needs `localhost` or HTTPS (microphone access is blocked on plain HTTP remote origins).
4. Press **Convert** — each file becomes a job with a live progress bar (`done/total` fragments).
5. When finished: play it inline, download the WAV, or delete the job. The history list (newest first) survives across submissions for the lifetime of the backend process.

The header dot shows backend health in real time. The GUI talks only to its own origin (`/api/*`), which Flask transparently proxies to FastAPI — no CORS setup and the backend never needs to be exposed to the network.

## Ports and URLs

| Deployment | GUI | API | Notes |
| --- | --- | --- | --- |
| `md-tts-web` defaults | `127.0.0.1:5000` | `127.0.0.1:5001` | Single process; API bound to localhost only. |
| `./start.sh local` | `$HOST:$PORT` (127.0.0.1:5000) | `127.0.0.1:$API_PORT` (8000) | |
| `./start.sh docker` | `localhost:$PORT` (5000) | `localhost:$API_PORT` (8000) | |
| `./start.sh hybrid` | `localhost:$PORT` (5000) | `localhost:$API_PORT` (8000) | Frontend container reaches the host backend via `host.docker.internal`. |

Interactive API docs (Swagger UI) are served by the backend at `http://<api-host>:<api-port>/docs`.

## REST API reference

Base URL: direct to the backend (`http://localhost:8000`) **or** through the GUI proxy (`http://localhost:5000` — same paths, same behavior).

| Method & path | Description |
| --- | --- |
| `GET /api/health` | Liveness check → `{"status": "ok"}` |
| `GET /api/engines` | Available engines + language support + capability metadata (`variants`, `speakers`, `supports_instruct`, `supports_voice_clone`, …). |
| `POST /api/jobs` | Create job(s). Multipart (batch) or JSON (single). |
| `GET /api/jobs` | All jobs, newest first. |
| `GET /api/jobs/{id}` | One job: state, progress, options, result URL. |
| `GET /api/jobs/{id}/audio` | The generated WAV (download/stream). |
| `DELETE /api/jobs/{id}` | Delete a job and its files from disk. |
| `POST /api/voice-references` | Upload a reference voice (multipart field `audio`, optional `ref_text`) → normalized mono 24 kHz WAV, reusable across jobs. |
| `GET /api/voice-references` | List uploaded reference voices. |
| `GET /api/voice-references/{id}/audio` | Stream a stored reference WAV. |
| `DELETE /api/voice-references/{id}` | Delete a stored reference voice. |

**Create a job from text (JSON):**

```bash
curl -X POST http://localhost:8000/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"name":"demo","content":"# Hola\n\nPrueba del sistema.","engine":"mms","language":"es","pause_ms":500,"workers":1}'
```

**Create a batch (multipart, one job per file):**

```bash
curl -X POST http://localhost:8000/api/jobs \
  -F files=@capitulo1.md -F files=@capitulo2.md \
  -F engine=kokoro -F language=es -F workers=2
```

**Poll status, download, delete:**

```bash
curl http://localhost:8000/api/jobs           # list / find the id
curl http://localhost:8000/api/jobs/<id>      # state + done/total
curl -OJ http://localhost:8000/api/jobs/<id>/audio
curl -X DELETE http://localhost:8000/api/jobs/<id>
```

**Qwen3-TTS jobs:**

```bash
# 1) Upload a reference voice for cloning (wav/mp3/ogg/webm, ≤ 20 MB)
curl -F "audio=@mi_voz.wav" -F "ref_text=Hola, esta es mi voz." \
  http://localhost:8000/api/voice-references
# → {"voice_reference":{"id":"1ea887973e4e","duration_s":3.0,"audio_url":"…"}}

# 2) Submit a clone job referencing it
curl -X POST http://localhost:8000/api/jobs -H 'Content-Type: application/json' \
  -d '{"name":"clon","content":"# Hola\n\nPrueba de voz clonada.","engine":"qwen3-clone",
       "language":"es","device":"auto","voice_ref_id":"1ea887973e4e"}'

# Voice design (no reference needed) and preset voices:
curl -X POST http://localhost:8000/api/jobs -H 'Content-Type: application/json' \
  -d '{"name":"design","content":"Hello world","engine":"qwen3-voicedesign","language":"en",
       "qwen_instruct":"Young male voice, tenor range, warm timbre"}'
curl -X POST http://localhost:8000/api/jobs -H 'Content-Type: application/json' \
  -d '{"name":"cv","content":"Hello world","engine":"qwen3-customvoice",
       "qwen_model":"0.6b","qwen_speaker":"Aiden"}'
```

**Job object:**

```json
{
  "id": "e9c57ee01cea",
  "name": "demo",
  "state": "done",                  // queued | running | done | error
  "done": 2,
  "total": 2,
  "error": null,
  "created_at": 1790631456.2,
  "duration": 17.7,
  "options": {"engine": "mms", "language": "es", "pause_ms": 500,
               "device": null, "workers": 1, "save_fragments": false},
  "audio_url": "/api/jobs/e9c57ee01cea/audio",
  "filename": "demo.wav"
}
```

Validation and limits:

| Field | Constraints |
| --- | --- |
| `engine` | must exist in the registry → else `400` |
| `language` | `es` \| `en` → else `400` |
| `device` | `auto` \| `cpu` \| `cuda` \| `cuda:N` \| empty → else `400` |
| `pause_ms` | 0 – 10000 → else `422` |
| `workers` | 1 – 16 → else `422` |
| `name` | 1 – 120 chars; sanitized for filenames |
| `content` | non-empty → else `422` |
| `qwen_model` | `1.7b` \| `0.6b` \| HuggingFace id / local path |
| `qwen_speaker` | must be a Qwen3 speaker (customvoice) → else `400` |
| `qwen_instruct` | required for `qwen3-voicedesign`; only `1.7b` customvoice supports it → else `400` |
| `voice_ref_id` | required for `qwen3-clone`, must exist → else `400` |
| `qwen_ref_text` | required for `qwen3-clone` unless `qwen_xvector_only=true` (falls back to the transcript stored with the reference) |
| uploaded files | ≤ 20 MB each; empty files are skipped |
| voice references | ≤ 20 MB, decodable audio (wav/mp3/ogg/webm/m4a; compressed formats need FFmpeg), ≥ 1 s |

Semantics:

- Jobs run **one at a time, FIFO** (models are heavy/GPU-bound). Parallelism *within* a job is controlled by `workers`.
- History and queue live **in memory**: restarting the backend clears the list (WAV files stay on disk in `--jobs-dir`).
- First job per engine pays the model download; the engine cache then keeps it loaded for the process lifetime.
- `GET .../audio` returns `409` until the job is `done`; `DELETE` returns `409` while running; unknown ids return `404`. Through the GUI proxy an unreachable backend surfaces as `502`.

## Docker

### CLI image (root `Dockerfile`)

```bash
docker build -t md-tts .
docker run --rm -v "$(pwd):/app" md-tts \
  --input-dir modificacion1 --output-dir output_audio \
  --engine kokoro --language en --save-fragments --workers 2
```

### Web stack (`docker/Dockerfile.backend` + `docker/Dockerfile.frontend`)

```bash
docker compose up --build        # or: ./start.sh docker
```

| Service | Image contents | Exposes |
| --- | --- | --- |
| `backend` | Full TTS stack (torch, transformers, kokoro, chatterbox, ffmpeg, espeak-ng, sox). Runs `uvicorn md_tts.api.app:app`. Build-arg `TTS_PROFILE=qwen` swaps chatterbox for the Qwen3 engines. | `8000` |
| `frontend` | **Lightweight**: only flask, requests and gunicorn (`PYTHONPATH=/app/src`, no ML deps). | `5000` |

Volumes and environment:

- `./output_audio:/app/output_audio` — generated audio lands on the host.
- `hf-cache` named volume — HuggingFace model cache survives image rebuilds.
- `MD_TTS_JOBS_DIR=/app/output_audio/web` — backend jobs directory.
- `MD_TTS_API_URL` — where the frontend proxies to (defaults to `http://backend:8000`; the hybrid mode overrides it with `http://host.docker.internal:<API_PORT>`).

GPU: uncomment `gpus: all` on the backend service (requires the NVIDIA Container Toolkit). On Linux, PyPI torch wheels are CUDA-enabled, so `--device cuda` then works end to end.

### Engine profiles in Docker

`chatterbox` and `qwen3-*` can't share an environment (incompatible `transformers` pins), so the images take a `TTS_PROFILE` build-arg:

```bash
# classic profile (default): kokoro + chatterbox
docker compose build

# qwen profile: qwen3-customvoice / qwen3-voicedesign / qwen3-clone
docker compose build --build-arg TTS_PROFILE=qwen          # both services
docker build --build-arg TTS_PROFILE=qwen -f docker/Dockerfile.backend -t md-tts-backend-qwen .
docker build --build-arg TTS_PROFILE=qwen -t md-tts-qwen .  # CLI image
```

## `start.sh` reference

```bash
./start.sh docker   # backend + frontend in Docker (foreground, Ctrl+C stops)
./start.sh local    # everything local with uv, single md-tts-web process
./start.sh hybrid   # frontend in Docker + backend on the host with uv
./start.sh stop     # kill the hybrid backend (PID file) + docker compose down
./start.sh help
```

Environment variables (all optional): `HOST` (default `127.0.0.1`, only `local`), `PORT` (GUI, default `5000`), `API_HOST` (default `0.0.0.0`, only `hybrid`), `API_PORT` (default `8000`), `TTS_PROFILE` (`qwen` → installs/exposes the Qwen3 engines instead of chatterbox; only `local`, `hybrid` and `docker`).

Hybrid-mode artifacts: the backend runs via `nohup uv run uvicorn md_tts.api.app:app`, with its PID in `$TMPDIR/md-tts/backend.pid` and logs in `$TMPDIR/md-tts/backend.log`. The script waits (up to 90 s) for the backend port to open before starting the frontend container.

Why hybrid? Model caches and downloads stay on the host (fast iteration, reusable HF cache), while the GUI is still a clean containerized unit.

## Configuration reference

| Variable | Used by | Default | Meaning |
| --- | --- | --- | --- |
| `MD_TTS_API_URL` | Flask frontend | `http://127.0.0.1:8001` | Backend base URL for the `/api/*` proxy. Set explicitly by compose and `md-tts-web` (flag `--api-url` wins over the env var). |
| `MD_TTS_JOBS_DIR` | FastAPI backend | `output_audio/web` | Root directory for job outputs (`<dir>/<job-id>/<name>.wav`). |
| `MD_TTS_VOICE_REFS_DIR` | FastAPI backend | `output_audio/voice_refs` | Where uploaded reference voices (voice cloning) are stored as normalized WAV + JSON metadata. Defaults relative to `MD_TTS_JOBS_DIR` or, when `--jobs-dir` is passed to `md-tts-web`, `<jobs-dir>/../voice_refs`. |
| `TTS_PROFILE` | `start.sh` | `classic` | Set to `qwen` to install/deploy the Qwen3 engines (`--extra qwen` / `--build-arg TTS_PROFILE=qwen`) instead of chatterbox-incompatible setups. |
| `HOST` / `PORT` / `API_HOST` / `API_PORT` | `start.sh` | `127.0.0.1` / `5000` / `0.0.0.0` / `8000` | Ports and bind addresses per mode (see above). |
| `HF_HOME` (optional) | engines | `~/.cache/huggingface` | Relocate the model cache (the compose backend mounts a volume at the default path). |

## Project structure

```text
src/md_tts/
  parser.py         # Markdown splitting by section/paragraph (file or text)
  tts.py            # TTS engines + ENGINE_REGISTRY + build_engine()
  audio_utils.py    # numpy<->AudioSegment, concatenation with pauses
  service.py        # Core service: engine cache, progress callbacks,
                    #   process_paragraph / process_markdown_text / _file
  cli.py            # `md-tts` CLI
  api/
    app.py          # FastAPI app + endpoints
    jobs.py         # JobManager: FIFO queue, worker thread, Job model
    voices.py       # VoiceRefStore: reference voices for qwen3-clone
  web/
    app.py          # Flask app: GUI + transparent /api/* proxy
    templates/index.html
    static/app.js   # i18n (ES/EN), polling, rendering
    static/style.css
  server_cli.py     # `md-tts-web` CLI (embedded uvicorn or frontend-only)
pyproject.toml      # deps + entry points
Dockerfile          # CLI container image
docker/
  Dockerfile.backend  # full TTS stack, runs uvicorn :8000
  Dockerfile.frontend # light GUI image, runs gunicorn :5000
docker-compose.yml
start.sh            # docker | local | hybrid | stop launcher
```

## Development notes

- **Layering:** `parser` → `service` → frontends (`cli`, `api`, `web`). The web layer must stay import-light: `md_tts.web` depends only on Flask/requests so the frontend image needs no ML stack.
- **Engine cache:** `service.EngineCache` keeps one engine instance per engine-specific key (see `ENGINE_CACHE_KEY_FIELDS` — e.g. `(device, qwen_model)` for Qwen3). Switching language, speaker or instruction on a cached Qwen3 engine does **not** reload the model: those are applied as runtime attributes (`_apply_runtime_options`), which is safe because jobs run one at a time. Used on every sequential path (CLI and web jobs). With `workers > 1` the pool forks separate processes and each builds its own engine (cache disabled).
- **Adding an engine:** subclass `TtsEngine` in `tts.py` and register it in `ENGINE_REGISTRY`. Add the name to `cli.py` choices and `service.LANGUAGE_AWARE_ENGINES` if it honors `--language` (plus `ENGINE_CACHE_KEY_FIELDS` if its cache key differs from `(language, device)` and `ENGINE_METADATA` in `tts.py` for UI capabilities). The API (`/api/engines`, validation) picks it up automatically.
- **GUI i18n:** translation dictionaries live at the top of `web/static/app.js`; the toggle persists via `localStorage` (`mdtts-lang`).
- **Job lifecycle:** `POST /api/jobs` → `queued` → `running` (progress callbacks update `done/total`) → `done` (with `audio_url`) or `error` (message surfaced in the GUI).

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `RuntimeWarning: Couldn't find ffmpeg or avconv` | Install FFmpeg (`apt-get install ffmpeg`). |
| Kokoro fails to phonemize | Install `espeak-ng`. |
| First job takes very long | Model download (GBs). In Docker, keep the `hf-cache` volume mounted so it happens once. |
| GUI shows *"API offline"* / red dot | Backend not running, or wrong `MD_TTS_API_URL`. Check `curl http://<api>/api/health`. |
| `502` from the GUI proxy | Backend unreachable from the frontend process (down, wrong URL, or firewall). |
| Hybrid mode: GUI up but backend "offline" | Backend must bind `0.0.0.0` (default `API_HOST`); the container reaches it via `host.docker.internal` (mapped by compose). |
| `address already in use` | Another process holds the port — change `PORT` / `API_PORT`. |
| History empty after restart | Expected: job state is in-memory. Files remain under `MD_TTS_JOBS_DIR`. |
| `409` on delete or audio download | Job still `running`; wait for `done`. |
| `413` on upload | File over the 20 MB limit — split it or paste text as JSON. |
| CUDA errors | `--device cuda` needs a GPU + CUDA-enabled torch; in Docker also `gpus: all` and the NVIDIA Container Toolkit. |
| `qwen-tts no está instalado` / `qwen-tts is not installed` | Install the optional extra: `uv sync --extra qwen` or `pip install 'md-tts[qwen]'`. |
| Browser 🎤 record button does nothing | Microphone access requires `localhost` or HTTPS; remote plain-HTTP origins block `getUserMedia`. |
| Reference audio upload rejected (`No se pudo decodificar`) | Install FFmpeg (needed for mp3/webm/m4a) and check the file is real audio ≥ 1 s. |

## Limitations

- Model downloads are large on the first execution of each engine (Qwen3 models: ~2–4 GB each).
- `vibevoice` and `cosyvoice` benefit strongly from GPU; Qwen3 1.7B runs comfortably on ~4 GB VRAM (bf16) and 0.6B on ~2 GB, but CPU also works (slower).
- `cosyvoice` requires its upstream repository and dependencies; `qwen3-*` requires the optional `qwen` extra.
- `vibevoice` currently behaves best in English; Spanish quality depends on prompt/content.
- Job history is per-process (no persistence); jobs run strictly one at a time.
- The Flask GUI runs on Werkzeug in local mode — fine for personal/LAN use, use the gunicorn frontend container for anything heavier.

## Model licenses

Check each model license before production use:

- facebook/mms-tts-spa / facebook/mms-tts-eng → CC-BY-NC 4.0 (non-commercial).
- hexgrad/Kokoro-82M → Apache 2.0.
- ResembleAI/chatterbox → MIT (with watermarking behavior).
- microsoft/VibeVoice-Realtime-0.5B → MIT.
- FunAudioLLM/Fun-CosyVoice3-0.5B-2512 → Apache 2.0.
- Qwen3-TTS (all models + tokenizer) → Apache 2.0. Note: cloning a real person's voice requires their consent.

---

# Markdown a audio (Español)

Convierte documentos Markdown en audio WAV (español/inglés) con motores TTS locales. El texto se divide por secciones (`#`) y párrafos; cada párrafo se sintetiza y los fragmentos se unen con una pausa configurable.

El proyecto ofrece tres interfaces sobre el mismo servicio: **CLI** (`md-tts`), **GUI web** (`md-tts-web`: Flask + FastAPI) y **librería Python** (`md_tts.service`).

## Motores

| Motor | Idiomas | Notas |
| --- | --- | --- |
| `mms` *(defecto)* | es, en | Ligero y fiable; funciona en CPU. |
| `kokoro` | es, en | Requiere `espeak-ng`. |
| `chatterbox` | es, en | Usa `language_id` si está el paquete multilingüe. |
| `vibevoice` | en | Optimizado para inglés; ignora `--language` (warning en español). |
| `cosyvoice` | es, en | Requiere instalar CosyVoice desde su repo; admite WAV de voz de referencia. |
| `qwen3-customvoice` | es, en | 9 voces premium + instrucción de estilo (solo 1.7B). Requiere el extra `qwen`. |
| `qwen3-voicedesign` | es, en | Diseña una voz nueva desde una descripción en texto (obligatoria). Requiere el extra `qwen`. |
| `qwen3-clone` | es, en | Clonación de voz desde audio de referencia (subida o grabación del navegador). Requiere el extra `qwen`. |

Notas Qwen3-TTS:

- **Dispositivo:** `--device auto` usa GPU si está disponible; bf16 en GPU, fp32 en CPU. Los 1.7B van holgados en ~4 GB de VRAM y los 0.6B en ~2 GB.
- **Idiomas reales de Qwen3:** 10 (es, en, zh, ja, ko, de, fr, ru, pt, it) — este proyecto solo expone es/en.
- **Guía completa de clonación y diseño de voz:** ver la sección inglesa «Voice cloning and voice design (Qwen3-TTS)».
- **Incompatibilidad:** los motores `qwen3-*` y `chatterbox` no pueden instalarse juntos (fijan versiones de `transformers` distintas). Los motores ausentes fallan con un mensaje claro de instalación.

## Instalación

```bash
uv sync                       # o: python3 -m pip install -e .
sudo apt-get install -y ffmpeg espeak-ng sox
# Para motores Qwen3-TTS (no compatibles con chatterbox en el mismo entorno):
uv sync --no-default-groups --extra qwen
```

Requiere Python 3.10–3.12. La primera ejecución de cada motor descarga pesos (puede ser GB).

## CLI

```bash
# Español con MMS (valores por defecto)
md-tts --input-dir modificacion1 --output-dir output_audio

# Inglés con MMS
md-tts --input-dir modificacion1 --output-dir output_audio_en --engine mms --language en

# Kokoro en GPU, guardando fragmentos (permite reanudar)
md-tts --engine kokoro --device cuda --save-fragments --language es

# Qwen3: voz premium + instrucción de estilo
md-tts --engine qwen3-customvoice --qwen-speaker Ryan --language en \
  --qwen-instruct "Very happy." --input-dir modificacion1 --output-dir out

# Qwen3: diseño de voz desde descripción
md-tts --engine qwen3-voicedesign --language en \
  --qwen-instruct "Voz masculina joven, tenor, timbre cálido, tono de documental" \
  --input-dir modificacion1 --output-dir out

# Qwen3: clonar voz desde audio de referencia (+ transcripción)
md-tts --engine qwen3-clone --language es --device auto \
  --qwen-ref-audio mi_voz.wav --qwen-ref-text "Hola, esta es mi voz." \
  --input-dir modificacion1 --output-dir out

# Cómo lo uso yo
uv sync
uv run md-tts --input-dir modificacion1 --output-dir output_audio --engine kokoro \
  --device cuda --pause-ms 500 --save-fragments --workers 10 --language es
```

Opciones principales: `--engine`, `--language` (es/en), `--pause-ms` (500), `--device` (cpu/cuda/auto), `--workers` (paralelismo por párrafo), `--save-fragments` (reanudación), `--cosyvoice-model-dir`, `--cosyvoice-prompt-wav`, `--qwen-model`, `--qwen-speaker`, `--qwen-instruct`, `--qwen-ref-audio`, `--qwen-ref-text`, `--qwen-xvector-only`. Ver `md-tts --help` y la sección inglesa «Voice cloning and voice design» para la guía completa.

## GUI web

Interfaz con carga por lotes o texto pegado, progreso en vivo, historial con reproductor/descarga/eliminar, e interfaz ES/EN. Con los motores `qwen3-*` aparecen opciones extra: tamaño de modelo, voz premium, instrucción de estilo, descripción de voz y clonación subiendo un mp3/wav o grabando desde el navegador (requiere localhost o HTTPS). El frontend Flask proxya `/api/*` al backend FastAPI (sin CORS, el backend no queda expuesto).

```bash
uv sync
uv run md-tts-web --host 127.0.0.1 --port 5000   # GUI :5000, API embebida :5001
```

O con `start.sh` (variables: `HOST`, `PORT`, `API_HOST`, `API_PORT`):

```bash
./start.sh docker   # backend + frontend en Docker (GUI :5000, API :8000)
./start.sh local    # todo local con uv (un proceso)
./start.sh hybrid   # frontend en Docker + backend local con uv
./start.sh stop     # detener backend híbrido y contenedores
```

- Imágenes en `docker/`: `Dockerfile.backend` (pila completa de TTS) y `Dockerfile.frontend` (ligera, sin torch).
- `docker-compose.yml` monta `./output_audio` (audio generado) y el volumen `hf-cache` (caché de modelos). Descomenta `gpus: all` para GPU.
- Perfiles de motor (los qwen3 y chatterbox no pueden convivir): `docker compose build --build-arg TTS_PROFILE=qwen` para incluir los motores Qwen3 en las imágenes.
- Documentación interactiva de la API en `http://<api>/docs`.

## API REST

| Método y ruta | Descripción |
| --- | --- |
| `GET /api/health` | Comprobación de estado. |
| `GET /api/engines` | Motores disponibles. |
| `POST /api/jobs` | Crear trabajo(s): multipart con `files` (lotes) o JSON `{name, content, …}`. |
| `GET /api/jobs` | Historial (más reciente primero). |
| `GET /api/jobs/{id}` | Estado y progreso (`done`/`total`). |
| `GET /api/jobs/{id}/audio` | Descargar el WAV generado. |
| `DELETE /api/jobs/{id}` | Eliminar trabajo y archivos. |
| `POST /api/voice-references` | Subir voz de referencia (multipart `audio`, `ref_text` opcional) para clonación. |
| `GET /api/voice-references` | Listar voces de referencia. |
| `GET /api/voice-references/{id}/audio` | Reproducir/descargar un WAV de referencia. |
| `DELETE /api/voice-references/{id}` | Eliminar voz de referencia. |

```bash
# Crear desde texto
curl -X POST http://localhost:8000/api/jobs -H 'Content-Type: application/json' \
  -d '{"name":"demo","content":"# Hola\n\nPrueba.","engine":"mms","language":"es"}'

# Lote por multipart
curl -X POST http://localhost:8000/api/jobs -F files=@cap1.md -F files=@cap2.md -F engine=kokoro

# Estado, descarga y borrado
curl http://localhost:8000/api/jobs/<id>
curl -OJ http://localhost:8000/api/jobs/<id>/audio
curl -X DELETE http://localhost:8000/api/jobs/<id>
```

Notas: los trabajos se procesan de uno en uno (FIFO); el historial vive en memoria y se reinicia con el proceso (los WAV quedan en disco); el primer trabajo por motor descarga el modelo y luego queda en caché; archivos ≤ 20 MB.

## Docker (CLI)

```bash
docker build -t md-tts .
docker run --rm -v "$(pwd):/app" md-tts --input-dir modificacion1 --output-dir output_audio --engine mms --language es
```

## Problemas frecuentes

| Síntoma | Solución |
| --- | --- |
| Warning de ffmpeg | `apt-get install ffmpeg`. |
| Kokoro no fonemiza | Instalar `espeak-ng`. |
| GUI en "API sin conexión" / 502 | Backend caído o `MD_TTS_API_URL` incorrecto; verifica `/api/health`. |
| Híbrido sin conexión al backend | El backend debe escuchar en `0.0.0.0` (`API_HOST` por defecto). |
| Puerto ocupado | Cambia `PORT`/`API_PORT`. |
| Historial vacío tras reiniciar | Esperado: el estado es en memoria. |
| 409 al descargar/borrar | El trabajo sigue en ejecución. |

## Licencias de modelos

- facebook/mms-tts-spa / facebook/mms-tts-eng → CC-BY-NC 4.0 (no comercial).
- hexgrad/Kokoro-82M → Apache 2.0.
- ResembleAI/chatterbox → MIT (con marca de agua).
- microsoft/VibeVoice-Realtime-0.5B → MIT.
- FunAudioLLM/Fun-CosyVoice3-0.5B-2512 → Apache 2.0.
- Qwen3-TTS (todos los modelos y el tokenizer) → Apache 2.0. Clonar la voz de una persona real requiere su consentimiento.
