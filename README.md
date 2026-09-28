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
                                                             CosyVoice)
        Browser ──► Flask GUI ──proxy /api/*──► FastAPI ──► JobManager (FIFO)
```

**Web GUI features:** drag & drop batch upload or pasted text · engine/language/pause/workers/device options · live per-job progress bars · job history with inline audio player, download and delete · bilingual interface (ES/EN) · backend health indicator.

## Table of contents

- [Supported engines](#supported-engines)
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

Language behavior:

- `--language` is applied by `mms`, `kokoro`, `chatterbox` (via `language_id`) and `cosyvoice` (language-specific prompt template).
- `vibevoice` is English-first; using `--language es` logs a warning.
- Default language is `es`.

## Requirements

| Requirement | Details |
| --- | --- |
| Python | 3.10 – 3.12 |
| FFmpeg | Required by `pydub` (audio concatenation/export) |
| espeak-ng | Only needed for the `kokoro` engine |
| Internet | First run per engine downloads model weights (can be GBs) |
| GPU (optional) | NVIDIA GPU + CUDA PyTorch for `--device cuda`; CPU works everywhere |
| Docker (optional) | For the containerized web stack; NVIDIA Container Toolkit for GPU |
| uv (optional) | Only needed for `./start.sh local` and `./start.sh hybrid` |

## Installation

```bash
git clone <repo-url> markdown-to-audio
cd markdown-to-audio
```

With **uv** (recommended — used by `start.sh`):

```bash
uv sync
```

With **pip**:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

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
| `--engine` | `mms` | `mms` · `kokoro` · `chatterbox` · `vibevoice` · `cosyvoice` |
| `--language` | `es` | `es` or `en` (engines that support it). |
| `--pause-ms` | `500` | Silence inserted between paragraphs, in milliseconds. |
| `--device` | auto | `cpu` or `cuda`. |
| `--workers` | `1` | Parallel paragraph synthesis via multiprocessing (`>1` disables the engine cache: each worker loads its own engine). |
| `--save-fragments` | off | Keep one WAV per paragraph under `output_dir/fragments/<name>/` — **re-running resumes** from existing fragments. |
| `--cosyvoice-model-dir` | download | Local CosyVoice model directory (skips the HF download). |
| `--cosyvoice-prompt-wav` | — | Reference voice WAV for CosyVoice. |

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
| `GET /api/engines` | Available engines + language support metadata. |
| `POST /api/jobs` | Create job(s). Multipart (batch) or JSON (single). |
| `GET /api/jobs` | All jobs, newest first. |
| `GET /api/jobs/{id}` | One job: state, progress, options, result URL. |
| `GET /api/jobs/{id}/audio` | The generated WAV (download/stream). |
| `DELETE /api/jobs/{id}` | Delete a job and its files from disk. |

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
| `device` | `cpu` \| `cuda` \| empty → else `400` |
| `pause_ms` | 0 – 10000 → else `422` |
| `workers` | 1 – 16 → else `422` |
| `name` | 1 – 120 chars; sanitized for filenames |
| `content` | non-empty → else `422` |
| uploaded files | ≤ 20 MB each; empty files are skipped |

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
| `backend` | Full TTS stack (torch, transformers, kokoro, chatterbox, ffmpeg, espeak-ng). Runs `uvicorn md_tts.api.app:app`. | `8000` |
| `frontend` | **Lightweight**: only flask, requests and gunicorn (`PYTHONPATH=/app/src`, no ML deps). | `5000` |

Volumes and environment:

- `./output_audio:/app/output_audio` — generated audio lands on the host.
- `hf-cache` named volume — HuggingFace model cache survives image rebuilds.
- `MD_TTS_JOBS_DIR=/app/output_audio/web` — backend jobs directory.
- `MD_TTS_API_URL` — where the frontend proxies to (defaults to `http://backend:8000`; the hybrid mode overrides it with `http://host.docker.internal:<API_PORT>`).

GPU: uncomment `gpus: all` on the backend service (requires the NVIDIA Container Toolkit). On Linux, PyPI torch wheels are CUDA-enabled, so `--device cuda` then works end to end.

## `start.sh` reference

```bash
./start.sh docker   # backend + frontend in Docker (foreground, Ctrl+C stops)
./start.sh local    # everything local with uv, single md-tts-web process
./start.sh hybrid   # frontend in Docker + backend on the host with uv
./start.sh stop     # kill the hybrid backend (PID file) + docker compose down
./start.sh help
```

Environment variables (all optional): `HOST` (default `127.0.0.1`, only `local`), `PORT` (GUI, default `5000`), `API_HOST` (default `0.0.0.0`, only `hybrid`), `API_PORT` (default `8000`).

Hybrid-mode artifacts: the backend runs via `nohup uv run uvicorn md_tts.api.app:app`, with its PID in `$TMPDIR/md-tts/backend.pid` and logs in `$TMPDIR/md-tts/backend.log`. The script waits (up to 90 s) for the backend port to open before starting the frontend container.

Why hybrid? Model caches and downloads stay on the host (fast iteration, reusable HF cache), while the GUI is still a clean containerized unit.

## Configuration reference

| Variable | Used by | Default | Meaning |
| --- | --- | --- | --- |
| `MD_TTS_API_URL` | Flask frontend | `http://127.0.0.1:8001` | Backend base URL for the `/api/*` proxy. Set explicitly by compose and `md-tts-web` (flag `--api-url` wins over the env var). |
| `MD_TTS_JOBS_DIR` | FastAPI backend | `output_audio/web` | Root directory for job outputs (`<dir>/<job-id>/<name>.wav`). |
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
- **Engine cache:** `service.EngineCache` keeps one engine instance per `(engine, language, device, cosyvoice paths)`. Used on every sequential path (CLI and web jobs). With `workers > 1` the pool forks separate processes and each builds its own engine (cache disabled).
- **Adding an engine:** subclass `TtsEngine` in `tts.py` and register it in `ENGINE_REGISTRY`. Add the name to `cli.py` choices and `service.LANGUAGE_AWARE_ENGINES` if it honors `--language`. The API (`/api/engines`, validation) picks it up automatically.
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

## Limitations

- Model downloads are large on the first execution of each engine.
- `vibevoice` and `cosyvoice` benefit strongly from GPU.
- `cosyvoice` requires its upstream repository and dependencies.
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

## Instalación

```bash
uv sync                       # o: python3 -m pip install -e .
sudo apt-get install -y ffmpeg espeak-ng
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

# Cómo lo uso yo
uv sync
uv run md-tts --input-dir modificacion1 --output-dir output_audio --engine kokoro \
  --device cuda --pause-ms 500 --save-fragments --workers 10 --language es
```

Opciones principales: `--engine`, `--language` (es/en), `--pause-ms` (500), `--device` (cpu/cuda), `--workers` (paralelismo por párrafo), `--save-fragments` (reanudación), `--cosyvoice-model-dir`, `--cosyvoice-prompt-wav`. Ver `md-tts --help`.

## GUI web

Interfaz con carga por lotes o texto pegado, progreso en vivo, historial con reproductor/descarga/eliminar, e interfaz ES/EN. El frontend Flask proxya `/api/*` al backend FastAPI (sin CORS, el backend no queda expuesto).

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
