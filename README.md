# Markdown to Audio (English / Spanish)

Convert all Markdown files in `modificacion1/` to WAV audio, splitting text by sections (`#`) and paragraphs. Each paragraph is synthesized with a TTS engine, then fragments are joined with a short pause to generate one audio file per Markdown.

## Supported Engines

- `mms` -> [facebook/mms-tts-spa](https://huggingface.co/facebook/mms-tts-spa) / [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng) (recommended).
- `kokoro` -> [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).
- `chatterbox` -> [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox).
- `vibevoice` -> [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B).
- `cosyvoice` -> [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) (requires CosyVoice installation from its repo).

Language behavior:
- `--language` is applied by `mms`, `kokoro`, `chatterbox` (`language_id`), and `cosyvoice` (language-specific prompt template).
- `vibevoice` is primarily optimized for English; using `--language es` shows a warning.

## Requirements

- Python 3.10+
- FFmpeg installed (`pydub` dependency)
- Internet connection the first time to download model weights

## Quick Install

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

Install FFmpeg on Debian/Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

## Usage

```bash
# Spanish with MMS (default language is es)
md-tts --input-dir modificacion1 --output-dir output_audio --engine mms

# English with MMS
md-tts --input-dir modificacion1 --output-dir output_audio_en --engine mms --language en

# Kokoro in Spanish
md-tts --engine kokoro --language es --input-dir modificacion1 --output-dir output_audio_kokoro_es

# Kokoro in English
md-tts --engine kokoro --language en --input-dir modificacion1 --output-dir output_audio_kokoro_en

# Chatterbox on GPU with language_id
md-tts --engine chatterbox --device cuda --language en

# Save paragraph fragments and process in parallel
md-tts --input-dir modificacion1 --output-dir output_audio --save-fragments --engine kokoro --language en --workers 2

# CosyVoice in Spanish (language-specific prompt template is applied)
md-tts --engine cosyvoice --language es --device cuda --cosyvoice-model-dir /path/to/cosyvoice_model

# How I use it
uv sync
uv run md-tts --input-dir modificacion1 --output-dir output_audio --engine kokoro --device cuda --pause-ms 500 --save-fragments --workers 10 --language es
```

Notes:
- Output WAV files are written to `--output-dir` with the same base name as each Markdown file.
- `--pause-ms` controls silence between paragraphs (default: `500`).
- `--device` accepts `cpu` or `cuda`.

## Docker (Linux)

```bash
docker build -t md-tts .
docker run --rm -v "$(pwd):/app" md-tts \
  --input-dir modificacion1 \
  --output-dir output_audio \
  --engine kokoro \
  --language en \
  --save-fragments \
  --workers 2
```

## Project Structure

```text
src/md_tts/
  parser.py       # Markdown splitting by section/paragraph
  tts.py          # TTS engines (MMS, Kokoro, Chatterbox, VibeVoice, CosyVoice)
  audio_utils.py  # Audio conversion and concatenation helpers
  cli.py          # CLI entrypoint
pyproject.toml    # Package metadata and dependencies
Dockerfile        # Container image
```

## Limitations

- Model downloads are large on first execution.
- `vibevoice` and some setups benefit strongly from GPU.
- `cosyvoice` requires its upstream repository and dependencies.
- `vibevoice` currently behaves best in English; Spanish quality depends on prompt/content and model behavior.

## Model Licenses

Check each model license before production use:
- facebook/mms-tts-spa / facebook/mms-tts-eng -> CC-BY-NC 4.0 (non-commercial).
- hexgrad/Kokoro-82M -> Apache 2.0.
- ResembleAI/chatterbox -> MIT (with watermarking behavior).
- microsoft/VibeVoice-Realtime-0.5B -> MIT.
- FunAudioLLM/Fun-CosyVoice3-0.5B-2512 -> Apache 2.0.

---

# Markdown a audio (Español)

Convierte los archivos Markdown de `modificacion1/` en audios WAV, separando por secciones y párrafos.

## Idioma

- `--language` funciona con `mms`, `kokoro`, `chatterbox` (vía `language_id`) y `cosyvoice` (vía plantilla de prompt por idioma).
- `vibevoice` está optimizado principalmente para inglés; al usar `--language es` se muestra warning.
- El valor por defecto es `es`.

## Uso rápido

```bash
# Español (por defecto)
md-tts --input-dir modificacion1 --output-dir output_audio --engine mms

# Inglés con MMS
md-tts --input-dir modificacion1 --output-dir output_audio_en --engine mms --language en

# Kokoro en inglés
md-tts --engine kokoro --language en --input-dir modificacion1 --output-dir output_audio_kokoro_en

# Cómo lo uso yo personalmente
uv sync
uv run md-tts --input-dir modificacion1 --output-dir output_audio --engine kokoro --device cuda --pause-ms 500 --save-fragments --workers 10 --language es
```

## Docker en Linux

```bash
docker build -t md-tts .
docker run --rm -v "$(pwd):/app" md-tts --input-dir modificacion1 --output-dir output_audio --engine mms --language es
```
