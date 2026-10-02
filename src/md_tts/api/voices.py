"""Almacén de audios de referencia para clonación de voz (engine qwen3-clone).

Los archivos subidos (wav/mp3/ogg/webm/m4a...) se normalizan a WAV mono 24 kHz
con pydub/ffmpeg y se guardan con un id de 12 hex junto a un JSON de metadatos
(transcripción opcional, duración, nombre original).

La API vive en ``/api/voice-references``; los jobs de clonación referencian el
audio por su ``id`` (``voice_ref_id``).
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("md_tts.api")

REF_ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")
TARGET_SAMPLE_RATE = 24000

#: Extensiones que pydub/ffmpeg saben decodificar (formato pasado a from_file).
KNOWN_AUDIO_FORMATS = {
    "wav", "mp3", "ogg", "webm", "m4a", "mp4", "aac", "flac", "wma", "opus", "aiff",
}

#: content-type -> formato pydub (fallback cuando no hay extensión conocida).
CONTENT_TYPE_FORMATS = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/vnd.wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/ogg": "ogg",
    "application/ogg": "ogg",
    "audio/webm": "webm",
    "video/webm": "webm",
    "audio/mp4": "mp4",
    "audio/x-m4a": "m4a",
    "audio/aac": "aac",
    "audio/flac": "flac",
}


def audio_format_hint(filename: str, content_type: str = "") -> Optional[str]:
    """Formato para pydub desde la extensión del archivo o su content-type."""

    suffix = Path(filename or "").suffix.lstrip(".").lower()
    if suffix in KNOWN_AUDIO_FORMATS:
        return suffix
    content_type = (content_type or "").split(";")[0].strip().lower()
    return CONTENT_TYPE_FORMATS.get(content_type)


def is_valid_ref_id(ref_id: str) -> bool:
    return bool(REF_ID_PATTERN.match(ref_id or ""))


class VoiceRefStore:
    """Guarda y consulta audios de referencia normalizados en ``base_dir``."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # -- rutas -------------------------------------------------------

    def _wav_path(self, ref_id: str) -> Path:
        return self.base_dir / f"{ref_id}.wav"

    def _meta_path(self, ref_id: str) -> Path:
        return self.base_dir / f"{ref_id}.json"

    # -- API pública -------------------------------------------------

    def save(
        self,
        raw: bytes,
        original_name: str = "reference",
        ref_text: Optional[str] = None,
        fmt: Optional[str] = None,
    ) -> dict:
        """Normaliza y guarda un audio subido. Retorna sus metadatos.

        ``fmt`` es el formato para pydub (extensión: wav, mp3, webm...); si es
        None se intenta autodetectar. Lanza ``ValueError`` si el audio no se
        puede decodificar o es demasiado corto (los formatos comprimidos
        requieren ffmpeg).
        """

        from pydub import AudioSegment

        try:
            segment = AudioSegment.from_file(io.BytesIO(raw), format=fmt)
        except Exception as exc:
            raise ValueError(
                f"No se pudo decodificar el audio ({exc}). Formatos: wav, mp3, ogg, webm, m4a "
                "(los comprimidos requieren ffmpeg instalado)."
            ) from exc

        segment = segment.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE).set_sample_width(2)
        if len(segment) < 1000:
            raise ValueError("El audio de referencia es demasiado corto (mínimo 1 s; ideal 3-10 s).")

        ref_id = uuid.uuid4().hex[:12]
        wav_path = self._wav_path(ref_id)
        segment.export(wav_path, format="wav")

        meta = {
            "id": ref_id,
            "original_name": Path(original_name or "reference").name[:120],
            "ref_text": (ref_text or "").strip() or None,
            "duration_s": round(len(segment) / 1000.0, 2),
            "sample_rate": TARGET_SAMPLE_RATE,
            "created_at": time.time(),
        }
        self._meta_path(ref_id).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        logger.info("Voice reference %s guardada (%.1f s, '%s')", ref_id, meta["duration_s"], meta["original_name"])
        return self._with_url(meta)

    def get(self, ref_id: str) -> Optional[dict]:
        if not is_valid_ref_id(ref_id):
            return None
        meta_path = self._meta_path(ref_id)
        if not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return self._with_url(meta)

    def list(self) -> List[dict]:
        refs = [self.get(path.stem) for path in self.base_dir.glob("*.json")]
        refs = [ref for ref in refs if ref]
        refs.sort(key=lambda ref: ref.get("created_at", 0), reverse=True)
        return refs

    def resolve_path(self, ref_id: str) -> Optional[Path]:
        """Ruta del WAV normalizado o None si no existe."""

        if not is_valid_ref_id(ref_id):
            return None
        wav = self._wav_path(ref_id)
        return wav if wav.is_file() else None

    def delete(self, ref_id: str) -> bool:
        if not is_valid_ref_id(ref_id):
            return False
        removed = False
        for path in (self._wav_path(ref_id), self._meta_path(ref_id)):
            if path.exists():
                path.unlink(missing_ok=True)
                removed = True
        if removed:
            logger.info("Voice reference %s eliminada", ref_id)
        return removed

    # -- helpers -----------------------------------------------------

    @staticmethod
    def _with_url(meta: dict) -> dict:
        meta = dict(meta)
        meta["audio_url"] = f"/api/voice-references/{meta.get('id')}/audio"
        return meta
