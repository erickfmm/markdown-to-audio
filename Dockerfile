FROM python:3.11-slim

# Perfil de motores: classic (defecto: kokoro + chatterbox), qwen (qwen3-*),
# kokoro (solo kokoro) o kokoro+qwen (kokoro + qwen3-*).
# chatterbox-tts y qwen-tts fijan versiones de transformers incompatibles y no
# pueden convivir en la misma imagen; elige el perfil al compilar:
#   docker build -t md-tts .                                       (classic)
#   docker build --build-arg TTS_PROFILE=qwen -t md-tts-qwen .     (qwen)
#   docker build --build-arg TTS_PROFILE=kokoro -t md-tts-kokoro . (kokoro)
#   docker build --build-arg TTS_PROFILE=kokoro+qwen -t md-tts-full . (kokoro+qwen)
ARG TTS_PROFILE=classic

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Optional mirror tweak for faster apt downloads
RUN sed -i 's|http://deb.debian.org|http://ftp.br.debian.org|g' /etc/apt/sources.list.d/debian.sources || \
    sed -i 's|http://deb.debian.org|http://ftp.br.debian.org|g' /etc/apt/sources.list

# System dependencies: ffmpeg for pydub, espeak-ng for Kokoro phonemization (en/es),
# sox for the qwen-tts package (qwen3-* engines)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg espeak-ng sox libsox-fmt-all && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN python3 -m pip install --upgrade pip

# Copy dependency manifest first to leverage Docker layer caching
COPY pyproject.toml /app/
# Base dependencies (engine ML packages installed per-profile below)
RUN python3 -m pip install --no-cache-dir \
    "transformers>=4.46.0" \
    "accelerate>=0.30.0" \
    "huggingface_hub>=0.24.0" \
    "torch>=2.1.0" \
    "torchaudio>=2.1.0" \
    "pydub>=0.25.1" \
    "soundfile>=0.12.1" \
    "numpy>=1.25.0" \
    "tqdm>=4.66.1"

# Copy source and install package in editable mode
COPY README.md /app/
COPY src /app/src
# Engine profile: classic = kokoro + chatterbox (transformers 4.46) /
# qwen = qwen3-* engines (transformers 4.57). Installed after `-e .` so pip
# settles each profile's exact pins without conflicts.
RUN python3 -m pip install --no-cache-dir -e . && \
    case "$TTS_PROFILE" in \
      qwen)        python3 -m pip install --no-cache-dir "qwen-tts>=0.1.1" "onnxruntime<1.24";; \
      kokoro)      python3 -m pip install --no-cache-dir "kokoro>=0.9.2";; \
      kokoro+qwen) python3 -m pip install --no-cache-dir "kokoro>=0.9.2" "qwen-tts>=0.1.1" "onnxruntime<1.24";; \
      *)           python3 -m pip install --no-cache-dir "kokoro>=0.9.2" "chatterbox-tts>=0.1.6";; \
    esac

ENTRYPOINT ["md-tts"]