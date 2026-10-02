#!/usr/bin/env bash
# start.sh - Lanza md-tts web (GUI Flask + backend FastAPI) en tres modos:
#
#   ./start.sh docker   Backend y frontend en Docker (docker compose).
#   ./start.sh local    Todo local con uv (un proceso: Flask + uvicorn embebido).
#   ./start.sh hybrid   Frontend en Docker, backend local con uv.
#   ./start.sh stop     Detiene el backend híbrido y los contenedores.
#
# Variables (con defaults):
#   HOST=0.0.0.0 PORT=5000 API_HOST=0.0.0.0 API_PORT=8000
#   TTS_PROFILE=classic|kokoro|qwen|kokoro+qwen   (motores a instalar)
set -euo pipefail

MODE="${1:-local}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5000}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
TTS_PROFILE="${TTS_PROFILE:-classic}"

RUN_DIR="${TMPDIR:-/tmp}/md-tts"
PID_FILE="$RUN_DIR/backend.pid"
LOG_FILE="$RUN_DIR/backend.log"

usage() {
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
}

compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
  else
    echo "Error: Docker Compose no está instalado." >&2
    exit 1
  fi
}

require_uv() {
  command -v uv >/dev/null 2>&1 || {
    echo "Error: uv no está instalado (https://docs.astral.sh/uv/)." >&2
    exit 1
  }
}

# uv sync según el perfil de motores:
#   classic      -> kokoro + chatterbox (grupo por defecto)
#   kokoro       -> solo kokoro
#   qwen         -> solo qwen3-*
#   kokoro+qwen  -> kokoro + qwen3-*
# (chatterbox-tts y qwen-tts fijan transformers incompatibles: nunca juntos)
sync_profile() {
  case "$TTS_PROFILE" in
    classic)      uv sync ;;
    kokoro)       uv sync --no-default-groups --extra kokoro ;;
    qwen)         uv sync --no-default-groups --extra qwen ;;
    kokoro+qwen)  uv sync --no-default-groups --extra kokoro --extra qwen ;;
    *)
      echo "Error: TTS_PROFILE inválido: '$TTS_PROFILE'. Use classic|kokoro|qwen|kokoro+qwen." >&2
      exit 1
      ;;
  esac
}

wait_port() {
  local host="$1" port="$2" tries="${3:-90}"
  for _ in $(seq 1 "$tries"); do
    if (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then
      exec 3>&- 3<&- 2>/dev/null || true
      return 0
    fi
    sleep 1
  done
  return 1
}

case "$MODE" in
  docker)
    echo ">> Modo Docker completo: GUI en http://localhost:${PORT} | API en http://localhost:${API_PORT}"
    # El perfil se pasa como build-arg a las imágenes (classic|kokoro|qwen|kokoro+qwen)
    TTS_PROFILE="$TTS_PROFILE" compose up --build
    ;;

  local)
    require_uv
    echo ">> Modo local (uv): GUI en http://${HOST}:${PORT} | API embebida en el puerto ${API_PORT} (perfil: ${TTS_PROFILE})"
    sync_profile
    # --no-sync: uv run re-sincronizaría con el grupo por defecto (classic)
    # y machacaría el perfil elegido arriba.
    exec uv run --no-sync md-tts-web \
      --host "$HOST" \
      --port "$PORT" \
      --api-host 127.0.0.1 \
      --api-port "$API_PORT"
    ;;

  hybrid)
    require_uv
    mkdir -p "$RUN_DIR"

    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Error: ya hay un backend híbrido corriendo (PID $(cat "$PID_FILE")). Usa './start.sh stop' primero." >&2
      exit 1
    fi

    echo ">> Modo híbrido: backend local (uv) + frontend Docker (perfil: ${TTS_PROFILE})"
    sync_profile

    echo ">> Iniciando backend FastAPI en ${API_HOST}:${API_PORT} (log: $LOG_FILE)"
    # --no-sync: el perfil ya está sincronizado arriba (sync_profile).
    nohup uv run --no-sync uvicorn md_tts.api.app:app \
      --host "$API_HOST" --port "$API_PORT" >"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"

    if ! wait_port 127.0.0.1 "$API_PORT"; then
      echo "Error: el backend no abrió el puerto ${API_PORT}. Revisa $LOG_FILE" >&2
      kill "$(cat "$PID_FILE")" 2>/dev/null || true
      rm -f "$PID_FILE"
      exit 1
    fi
    echo ">> Backend listo (PID $(cat "$PID_FILE"))."

    echo ">> Iniciando frontend Docker en http://localhost:${PORT}"
    MD_TTS_API_URL="http://host.docker.internal:${API_PORT}" \
      compose up --build -d --no-deps frontend

    echo ""
    echo "Listo:"
    echo "  GUI:  http://localhost:${PORT}"
    echo "  API:  http://localhost:${API_PORT}/api/health"
    echo "  Log:  $LOG_FILE"
    echo "  Stop: ./start.sh stop"
    ;;

  stop)
    if [[ -f "$PID_FILE" ]]; then
      PID="$(cat "$PID_FILE")"
      if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" && echo ">> Backend híbrido detenido (PID $PID)."
      fi
      rm -f "$PID_FILE"
    fi
    compose down 2>/dev/null || true
    echo ">> Contenedores detenidos."
    ;;

  help|-h|--help)
    usage
    ;;

  *)
    usage
    echo "Error: modo desconocido: $MODE" >&2
    exit 1
    ;;
esac
