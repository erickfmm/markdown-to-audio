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
from .tts import build_engine

logger = logging.getLogger("md_tts")

LANGUAGE_AWARE_ENGINES = {"mms", "kokoro", "chatterbox", "cosyvoice"}

ProgressCallback = Callable[[int, int], None]


class EngineCache:
    """Caché thread-safe de motores TTS por (engine, language, device, ...).

    En un servidor de larga vida evita recargar el modelo en cada párrafo.
    """

    def __init__(self) -> None:
        self._engines: dict = {}
        self._lock = threading.Lock()

    def get(self, engine_name: str, **kwargs):
        key = (
            engine_name,
            kwargs.get("language"),
            kwargs.get("device"),
            str(kwargs.get("cosyvoice_model_dir") or ""),
            str(kwargs.get("cosyvoice_prompt_wav") or ""),
        )
        with self._lock:
            engine = self._engines.get(key)
            if engine is None:
                engine = build_engine(engine_name, **kwargs)
                self._engines[key] = engine
            return engine

    def clear(self) -> None:
        with self._lock:
            self._engines.clear()


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
    try:
        if use_cache:
            engine = _engine_cache.get(
                engine_name,
                language=language,
                device=device,
                cosyvoice_model_dir=cosyvoice_model_dir,
                cosyvoice_prompt_wav=cosyvoice_prompt_wav,
            )
        else:
            engine = build_engine(
                engine_name,
                language=language,
                device=device,
                cosyvoice_model_dir=cosyvoice_model_dir,
                cosyvoice_prompt_wav=cosyvoice_prompt_wav,
            )
        seg = engine.synthesize(text)

        # Guardar fragmento si se solicita
        if temp_dir:
            fragment_path = temp_dir / fragment_name
            seg.export(fragment_path, format="wav")

        return (para.index, seg)
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
        save_fragments=save_fragments,
        workers=workers,
        progress_callback=progress_callback,
    )
