"""Servicio principal: conversión de Markdown a audio, reutilizable por CLI y API.

Contiene la lógica extraída del CLI original (procesamiento por párrafo y por
documento) junto con mejoras para ejecución como servidor:

- :class:`EngineCache`: caché thread-safe de motores TTS para cargar modelos
  una sola vez por combinación (engine, idioma, dispositivo).
- ``progress_callback``: notifica avances (fragmentos hechos / total).
- :func:`process_markdown_text`: convierte texto Markdown directo (sin archivo).
"""

from __future__ import annotations

import logging
import re
import threading
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from pydub import AudioSegment
from tqdm import tqdm

from . import parser as md_parser
from .audio_utils import concatenate_with_pause
from .tts import build_engine, resolve_qwen_model_id

logger = logging.getLogger("md_tts")

LANGUAGE_AWARE_ENGINES = {
    "mms",
    "kokoro",
    "chatterbox",
    "cosyvoice",
    "qwen3-customvoice",
    "qwen3-voicedesign",
    "qwen3-clone",
}

#: Familia (clave en QWEN_MODEL_ALIASES) de cada engine Qwen3.
QWEN_ENGINE_VARIANTS = {
    "qwen3-customvoice": "customvoice",
    "qwen3-voicedesign": "voicedesign",
    "qwen3-clone": "base",
}

#: Campos que definen la clave de la caché por engine. Los engines Qwen3 NO
#: incluyen language/estilo: cambiar idioma, speaker o instruct NO recarga el
#: modelo (esos valores se aplican como atributos de ejecución, ver
#: ``_apply_runtime_options``).
ENGINE_CACHE_KEY_FIELDS = {
    "mms": ("language", "device"),
    "kokoro": ("language", "device"),
    "chatterbox": ("language", "device"),
    "vibevoice": ("device",),
    "cosyvoice": ("language", "device", "cosyvoice_model_dir", "cosyvoice_prompt_wav"),
    "qwen3-customvoice": ("device", "qwen_model"),
    "qwen3-voicedesign": ("device", "qwen_model"),
    "qwen3-clone": ("device", "qwen_model"),
}
DEFAULT_CACHE_KEY_FIELDS = ("language", "device")

#: Opciones que se aplican al engine obtenido de la caché antes de sintetizar
#: (seguro: los jobs se procesan de uno en uno en el worker FIFO y el CLI es
#: monohilo).
RUNTIME_OPTION_FIELDS = (
    "language",
    "qwen_speaker",
    "qwen_instruct",
    "qwen_ref_audio",
    "qwen_ref_text",
    "qwen_xvector_only",
)

ProgressCallback = Callable[[int, int], None]


class EngineCache:
    """Caché thread-safe de motores TTS por (engine, idioma, dispositivo, ...).

    En un servidor de larga vida evita recargar el modelo en cada párrafo.
    """

    def __init__(self) -> None:
        self._engines: dict = {}
        self._lock = threading.Lock()

    def get(self, engine_name: str, **kwargs):
        key = _engine_cache_key(engine_name, kwargs)
        with self._lock:
            engine = self._engines.get(key)
            if engine is None:
                engine = build_engine(engine_name, **kwargs)
                self._engines[key] = engine
            # Aplica idioma/estilo SIEMPRE: cambiar speaker/instruct/referencia
            # sobre un engine cacheado no reconstruye el modelo.
            _apply_runtime_options(engine, kwargs)
            return engine

    def clear(self) -> None:
        with self._lock:
            self._engines.clear()


def _engine_cache_key(engine_name: str, kwargs: dict) -> tuple:
    fields = ENGINE_CACHE_KEY_FIELDS.get(engine_name, DEFAULT_CACHE_KEY_FIELDS)
    values = []
    for field in fields:
        value = kwargs.get(field)
        if field == "qwen_model" and engine_name in QWEN_ENGINE_VARIANTS:
            # Normaliza alias (1.7b) vs id completo para no cargar dos veces
            # el mismo modelo.
            value = resolve_qwen_model_id(QWEN_ENGINE_VARIANTS[engine_name], value)
        values.append(str(value or ""))
    return (engine_name, *values)


def _apply_runtime_options(engine, kwargs: dict) -> None:
    """Aplica idioma/estilo a un engine (posiblemente reutilizado de la caché)."""

    for field in RUNTIME_OPTION_FIELDS:
        value = kwargs.get(field)
        if value is not None and hasattr(engine, field):
            setattr(engine, field, value)


_engine_cache = EngineCache()


def sanitize_name(name: str, fallback: str = "documento") -> str:
    """Sanitiza un nombre para usarlo como nombre de archivo de salida."""

    safe = re.sub(r"[^\w\-. ]+", "_", name.strip())[:60].strip(" ._")
    return safe or fallback


def process_paragraph(
    para: md_parser.Paragraph,
    engine_name: str,
    language: str,
    device: Optional[str] = None,
    cosyvoice_model_dir: Optional[Path] = None,
    cosyvoice_prompt_wav: Optional[Path] = None,
    qwen_model: Optional[str] = None,
    qwen_speaker: Optional[str] = None,
    qwen_instruct: Optional[str] = None,
    qwen_ref_audio: Optional[str] = None,
    qwen_ref_text: Optional[str] = None,
    qwen_xvector_only: bool = False,
    temp_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> Tuple[int, Optional[AudioSegment]]:
    """Procesa un párrafo individual. Retorna (index, segment)."""
    text = para.text.strip()
    if not text:
        return (para.index, None)

    fragment_name = f"{para.index:04d}_{para.section[:30] if para.section else 'intro'}.wav"

    # Verificar si el fragmento ya existe (reanudar)
    if temp_dir:
        fragment_path = temp_dir / fragment_name
        if fragment_path.exists():
            try:
                seg = AudioSegment.from_wav(fragment_path)
                return (para.index, seg)
            except Exception as exc:
                logger.warning("Error cargando fragmento existente %s: %s", fragment_path, exc)

    # Sintetizar el fragmento
    engine_kwargs = {
        "language": language,
        "device": device,
        "cosyvoice_model_dir": cosyvoice_model_dir,
        "cosyvoice_prompt_wav": cosyvoice_prompt_wav,
        "qwen_model": qwen_model,
        "qwen_speaker": qwen_speaker,
        "qwen_instruct": qwen_instruct,
        "qwen_ref_audio": qwen_ref_audio,
        "qwen_ref_text": qwen_ref_text,
        "qwen_xvector_only": qwen_xvector_only,
    }
    try:
        if use_cache:
            # EngineCache aplica idioma/estilo al engine cacheado y reconstruye
            # el modelo solo cuando cambia la clave (device, qwen_model, ...).
            engine = _engine_cache.get(engine_name, **engine_kwargs)
        else:
            engine = build_engine(engine_name, **engine_kwargs)
        seg = engine.synthesize(text)

        # Guardar fragmento si se solicita
        if temp_dir:
            fragment_path = temp_dir / fragment_name
            seg.export(fragment_path, format="wav")

        return (para.index, seg)
    except ImportError:
        # Dependencias ausentes (qwen-tts, CosyVoice...): abortar con el
        # mensaje de instalación claro en vez de fallar párrafo a párrafo.
        raise
    except Exception as exc:
        logger.error("Error sintetizando párrafo %s: %s", para.index, exc)
        return (para.index, None)


def process_markdown_text(
    name: str,
    content: str,
    output_dir: Path,
    engine_name: str,
    language: str,
    pause_ms: int,
    device: Optional[str] = None,
    cosyvoice_model_dir: Optional[Path] = None,
    cosyvoice_prompt_wav: Optional[Path] = None,
    qwen_model: Optional[str] = None,
    qwen_speaker: Optional[str] = None,
    qwen_instruct: Optional[str] = None,
    qwen_ref_audio: Optional[str] = None,
    qwen_ref_text: Optional[str] = None,
    qwen_xvector_only: bool = False,
    save_fragments: bool = False,
    workers: int = 1,
    progress_callback: Optional[ProgressCallback] = None,
    show_progress: bool = True,
    use_cache: bool = True,
) -> Path:
    """Convierte texto Markdown a un WAV y retorna la ruta de salida.

    ``progress_callback(done, total)`` se invoca al inicio (0, total) y tras
    cada fragmento completado.
    """
    paragraphs = md_parser.split_markdown_text(content)
    total = len(paragraphs)
    logger.info("%s -> %d fragmentos", name, total)
    if progress_callback:
        progress_callback(0, total)

    # Crear carpeta para fragmentos si se solicita
    temp_dir = None
    if save_fragments:
        temp_dir = output_dir / "fragments" / sanitize_name(name)
        temp_dir.mkdir(parents=True, exist_ok=True)

        existing_fragments = list(temp_dir.glob("*.wav"))
        if existing_fragments:
            logger.info("Encontrados %d fragmentos existentes, reanudando...", len(existing_fragments))

    segments: List[Tuple[int, AudioSegment]] = []

    if workers > 1:
        # Procesamiento paralelo (sin caché: cada subprocess construye su engine)
        process_func = partial(
            process_paragraph,
            engine_name=engine_name,
            language=language,
            device=device,
            cosyvoice_model_dir=cosyvoice_model_dir,
            cosyvoice_prompt_wav=cosyvoice_prompt_wav,
            qwen_model=qwen_model,
            qwen_speaker=qwen_speaker,
            qwen_instruct=qwen_instruct,
            qwen_ref_audio=qwen_ref_audio,
            qwen_ref_text=qwen_ref_text,
            qwen_xvector_only=qwen_xvector_only,
            temp_dir=temp_dir,
            use_cache=False,
        )

        with Pool(processes=workers) as pool:
            results_iter = pool.imap(process_func, paragraphs)
            if show_progress:
                results_iter = tqdm(results_iter, desc=name, unit="p", total=total)
            for done, result in enumerate(results_iter, start=1):
                if result[1] is not None:
                    segments.append(result)
                if progress_callback:
                    progress_callback(done, total)
    else:
        # Procesamiento secuencial (con caché de engines)
        iterator = paragraphs
        if show_progress:
            iterator = tqdm(paragraphs, desc=name, unit="p")  # type: ignore[assignment]
        for done, para in enumerate(iterator, start=1):
            idx, seg = process_paragraph(
                para,
                engine_name=engine_name,
                language=language,
                device=device,
                cosyvoice_model_dir=cosyvoice_model_dir,
                cosyvoice_prompt_wav=cosyvoice_prompt_wav,
                qwen_model=qwen_model,
                qwen_speaker=qwen_speaker,
                qwen_instruct=qwen_instruct,
                qwen_ref_audio=qwen_ref_audio,
                qwen_ref_text=qwen_ref_text,
                qwen_xvector_only=qwen_xvector_only,
                temp_dir=temp_dir,
                use_cache=use_cache,
            )
            if seg is not None:
                segments.append((idx, seg))
            if progress_callback:
                progress_callback(done, total)

    if not segments:
        raise RuntimeError(f"No se generaron segmentos de audio para {name}")

    # Ordenar segmentos por índice para mantener el orden correcto
    segments.sort(key=lambda x: x[0])
    audio_segments = [seg for _, seg in segments]

    combined = concatenate_with_pause(audio_segments, pause_ms=pause_ms)
    output_path = output_dir / f"{sanitize_name(name)}.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path, format="wav")
    return output_path


def process_markdown_file(
    md_file: Path,
    output_dir: Path,
    engine_name: str,
    language: str,
    pause_ms: int,
    device: Optional[str] = None,
    cosyvoice_model_dir: Optional[Path] = None,
    cosyvoice_prompt_wav: Optional[Path] = None,
    qwen_model: Optional[str] = None,
    qwen_speaker: Optional[str] = None,
    qwen_instruct: Optional[str] = None,
    qwen_ref_audio: Optional[str] = None,
    qwen_ref_text: Optional[str] = None,
    qwen_xvector_only: bool = False,
    save_fragments: bool = False,
    workers: int = 1,
    progress_callback: Optional[ProgressCallback] = None,
) -> Path:
    """Procesa un archivo .md del disco (compatibilidad con el CLI)."""

    content = md_file.read_text(encoding="utf-8")
    return process_markdown_text(
        name=md_file.stem,
        content=content,
        output_dir=output_dir,
        engine_name=engine_name,
        language=language,
        pause_ms=pause_ms,
        device=device,
        cosyvoice_model_dir=cosyvoice_model_dir,
        cosyvoice_prompt_wav=cosyvoice_prompt_wav,
        qwen_model=qwen_model,
        qwen_speaker=qwen_speaker,
        qwen_instruct=qwen_instruct,
        qwen_ref_audio=qwen_ref_audio,
        qwen_ref_text=qwen_ref_text,
        qwen_xvector_only=qwen_xvector_only,
        save_fragments=save_fragments,
        workers=workers,
        progress_callback=progress_callback,
    )
