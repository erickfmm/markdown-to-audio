"""App FastAPI con los endpoints de trabajos, metadatos y voces de referencia."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError

from .. import service
from ..tts import (
    ENGINE_METADATA,
    ENGINE_REGISTRY,
    QWEN_SPEAKERS,
    resolve_qwen_model_id,
)
from .jobs import JobManager, JobOptions, JobState
from .voices import VoiceRefStore, audio_format_hint

logger = logging.getLogger("md_tts.api")

DEFAULT_JOBS_DIR = Path(os.environ.get("MD_TTS_JOBS_DIR", "output_audio/web"))
DEFAULT_VOICE_REFS_DIR = Path(
    os.environ.get("MD_TTS_VOICE_REFS_DIR", str(DEFAULT_JOBS_DIR.parent / "voice_refs"))
)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

#: Valores por defecto de los metadatos por engine (ver ENGINE_METADATA en tts.py).
ENGINE_METADATA_DEFAULTS = {
    "variants": [],
    "default_variant": None,
    "speakers": [],
    "default_speaker": None,
    "supports_instruct": False,
    "requires_instruct": False,
    "supports_voice_clone": False,
    "requires_voice_reference": False,
}

#: Familia Qwen3 (clave en QWEN_MODEL_ALIASES) por engine.
QWEN_ENGINE_VARIANTS = dict(service.QWEN_ENGINE_VARIANTS)


class JobCreateRequest(BaseModel):
    name: str = Field(default="documento", min_length=1, max_length=120)
    content: str = Field(min_length=1)
    engine: str = "mms"
    language: str = "es"
    pause_ms: int = Field(default=500, ge=0, le=10000)
    device: Optional[str] = None
    workers: int = Field(default=1, ge=1, le=16)
    save_fragments: bool = False
    qwen_model: Optional[str] = None
    qwen_speaker: Optional[str] = None
    qwen_instruct: Optional[str] = None
    voice_ref_id: Optional[str] = None
    qwen_ref_text: Optional[str] = None
    qwen_xvector_only: bool = False


def _validate_device(device) -> None:
    device_clean = str(device or "").strip().lower()
    valid = device_clean in ("", "auto", "cpu", "cuda") or device_clean.startswith("cuda:")
    if not valid:
        raise HTTPException(
            status_code=400,
            detail=f"Device inválido: {device}. Use 'auto', 'cpu' o 'cuda'.",
        )


def _validate_qwen_options(
    engine: str,
    voice_store: Optional[VoiceRefStore],
    qwen_model: Optional[str],
    qwen_speaker: Optional[str],
    qwen_instruct: Optional[str],
    voice_ref_id: Optional[str],
    qwen_ref_text: Optional[str],
    qwen_xvector_only: bool,
) -> tuple:
    """Valida las opciones qwen3-* y resuelve la referencia de voz.

    Retorna ``(ref_path, ref_text)``: ruta del WAV de referencia y transcripción
    (posiblemente resuelta desde los metadatos de la referencia) o valores
    vacíos si no aplican.
    """

    if engine not in QWEN_ENGINE_VARIANTS:
        return ("", None)
    model_id = resolve_qwen_model_id(QWEN_ENGINE_VARIANTS[engine], qwen_model)

    if engine == "qwen3-customvoice":
        if qwen_speaker and qwen_speaker not in QWEN_SPEAKERS:
            raise HTTPException(
                status_code=400,
                detail=f"Speaker desconocido: {qwen_speaker}. Opciones: {list(QWEN_SPEAKERS)}",
            )
        if (qwen_instruct or "").strip() and "0.6B" in model_id:
            raise HTTPException(
                status_code=400,
                detail="El modelo 0.6B-CustomVoice no soporta instruct; use qwen_model=1.7b.",
            )
        return ("", None)

    if engine == "qwen3-voicedesign":
        if not (qwen_instruct or "").strip():
            raise HTTPException(
                status_code=400,
                detail="qwen3-voicedesign requiere qwen_instruct: descripción de la voz "
                "(género, edad, timbre, emoción, ritmo).",
            )
        return ("", None)

    # qwen3-clone: requiere referencia de voz subida a /api/voice-references.
    if not voice_ref_id:
        raise HTTPException(
            status_code=400,
            detail="qwen3-clone requiere voice_ref_id (suba el audio a /api/voice-references).",
        )
    if voice_store is None:
        raise HTTPException(status_code=500, detail="Almacén de voces no disponible.")
    ref_path = voice_store.resolve_path(voice_ref_id)
    if ref_path is None:
        raise HTTPException(status_code=400, detail=f"Voice reference no encontrada: {voice_ref_id}")
    # La transcripción puede guardarse junto a la referencia al subirla.
    if not (qwen_ref_text or "").strip():
        meta = voice_store.get(voice_ref_id) or {}
        qwen_ref_text = meta.get("ref_text") or None
    if not (qwen_ref_text or "").strip() and not qwen_xvector_only:
        raise HTTPException(
            status_code=400,
            detail="qwen3-clone requiere qwen_ref_text (transcripción del audio) o "
            "qwen_xvector_only=true.",
        )
    return (str(ref_path), qwen_ref_text)


def _parse_options(
    engine: str,
    language: str,
    pause_ms,
    device,
    workers,
    save_fragments,
    voice_store: Optional[VoiceRefStore] = None,
    qwen_model: Optional[str] = None,
    qwen_speaker: Optional[str] = None,
    qwen_instruct: Optional[str] = None,
    voice_ref_id: Optional[str] = None,
    qwen_ref_text: Optional[str] = None,
    qwen_xvector_only: bool = False,
) -> JobOptions:
    if engine not in ENGINE_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Engine desconocido: {engine}. Opciones: {list(ENGINE_REGISTRY)}",
        )
    if language not in ("es", "en"):
        raise HTTPException(status_code=400, detail=f"Idioma inválido: {language}. Use 'es' o 'en'.")
    _validate_device(device)

    qwen_ref_audio, qwen_ref_text = _validate_qwen_options(
        engine,
        voice_store,
        qwen_model,
        qwen_speaker,
        qwen_instruct,
        voice_ref_id,
        qwen_ref_text,
        qwen_xvector_only,
    )

    try:
        return JobOptions.from_values(
            engine=engine,
            language=language,
            pause_ms=pause_ms,
            device=device,
            workers=workers,
            save_fragments=save_fragments,
            qwen_model=qwen_model,
            qwen_speaker=qwen_speaker,
            qwen_instruct=qwen_instruct,
            qwen_ref_audio=qwen_ref_audio or None,
            qwen_ref_text=qwen_ref_text,
            qwen_xvector_only=qwen_xvector_only,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def create_app(
    jobs_dir: Optional[Path] = None,
    voice_refs_dir: Optional[Path] = None,
) -> FastAPI:
    app = FastAPI(
        title="md-tts",
        description="API para convertir Markdown a audio mediante trabajos asíncronos.",
        version="0.1.0",
    )
    manager = JobManager(jobs_dir or DEFAULT_JOBS_DIR)
    voice_store = VoiceRefStore(voice_refs_dir or DEFAULT_VOICE_REFS_DIR)
    app.state.jobs = manager
    app.state.voice_refs = voice_store

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/engines")
    def engines() -> dict:
        payload = []
        for name in ENGINE_REGISTRY:
            meta = dict(ENGINE_METADATA_DEFAULTS)
            meta.update(ENGINE_METADATA.get(name, {}))
            payload.append(
                {
                    "id": name,
                    "language_aware": name in service.LANGUAGE_AWARE_ENGINES,
                    **meta,
                }
            )
        return {"engines": payload, "languages": ["es", "en"]}

    # -- Voces de referencia (clonación) ----------------------------

    @app.post("/api/voice-references")
    async def create_voice_reference(request: Request) -> dict:
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("multipart/form-data"):
            raise HTTPException(status_code=400, detail="Use multipart/form-data con el campo 'audio'.")
        form = await request.form()
        upload = form.get("audio")
        if upload is None or isinstance(upload, str):
            raise HTTPException(status_code=400, detail="Sin archivo: use el campo 'audio'.")
        raw = await upload.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Archivo vacío.")
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Archivo demasiado grande (máx. 20 MB).")
        ref_text = form.get("ref_text")
        try:
            meta = voice_store.save(
                raw,
                original_name=getattr(upload, "filename", None) or "reference",
                ref_text=(str(ref_text) if ref_text else None),
                fmt=audio_format_hint(
                    getattr(upload, "filename", None) or "",
                    getattr(upload, "content_type", None) or "",
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"voice_reference": meta}

    @app.get("/api/voice-references")
    def list_voice_references() -> dict:
        return {"voice_references": voice_store.list()}

    @app.get("/api/voice-references/{ref_id}/audio")
    def get_voice_reference_audio(ref_id: str) -> FileResponse:
        path = voice_store.resolve_path(ref_id)
        if path is None:
            raise HTTPException(status_code=404, detail=f"Voice reference no encontrada: {ref_id}")
        return FileResponse(path, media_type="audio/wav", filename=path.name)

    @app.delete("/api/voice-references/{ref_id}")
    def delete_voice_reference(ref_id: str) -> dict:
        if not voice_store.delete(ref_id):
            raise HTTPException(status_code=404, detail=f"Voice reference no encontrada: {ref_id}")
        return {"deleted": ref_id}

    # -- Trabajos ----------------------------------------------------

    @app.post("/api/jobs")
    async def create_jobs(request: Request) -> dict:
        content_type = request.headers.get("content-type", "")
        created = []

        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            uploads = form.getlist("files")
            if not uploads:
                raise HTTPException(status_code=400, detail="Sin archivos: use el campo 'files'.")
            options = _parse_options(
                engine=form.get("engine") or "mms",
                language=form.get("language") or "es",
                pause_ms=form.get("pause_ms") or 500,
                device=form.get("device") or None,
                workers=form.get("workers") or 1,
                save_fragments=form.get("save_fragments") or False,
                voice_store=voice_store,
                qwen_model=form.get("qwen_model") or None,
                qwen_speaker=form.get("qwen_speaker") or None,
                qwen_instruct=form.get("qwen_instruct") or None,
                voice_ref_id=form.get("voice_ref_id") or None,
                qwen_ref_text=form.get("qwen_ref_text") or None,
                qwen_xvector_only=form.get("qwen_xvector_only") or False,
            )
            for upload in uploads:
                filename = getattr(upload, "filename", None)
                if not filename:
                    continue
                raw = await upload.read()
                if len(raw) > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"Archivo demasiado grande: {filename} (máx. 20 MB).")
                if not raw.strip():
                    continue
                name = Path(filename).stem or "documento"
                created.append(
                    manager.submit(name=name, content=raw.decode("utf-8", errors="replace"), options=options)
                )
        else:
            try:
                body = await request.json()
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Cuerpo JSON inválido.") from exc
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="Se esperaba un objeto JSON.")
            try:
                req = JobCreateRequest(**body)
            except ValidationError as exc:
                raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc
            options = _parse_options(
                engine=req.engine,
                language=req.language,
                pause_ms=req.pause_ms,
                device=req.device,
                workers=req.workers,
                save_fragments=req.save_fragments,
                voice_store=voice_store,
                qwen_model=req.qwen_model,
                qwen_speaker=req.qwen_speaker,
                qwen_instruct=req.qwen_instruct,
                voice_ref_id=req.voice_ref_id,
                qwen_ref_text=req.qwen_ref_text,
                qwen_xvector_only=req.qwen_xvector_only,
            )
            created.append(manager.submit(name=req.name, content=req.content, options=options))

        if not created:
            raise HTTPException(status_code=400, detail="Nada que convertir (archivos vacíos).")
        return {"jobs": [job.to_dict() for job in created]}

    @app.get("/api/jobs")
    def list_jobs() -> dict:
        return {"jobs": [job.to_dict() for job in manager.list_jobs()]}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job no encontrado: {job_id}")
        return job.to_dict()

    @app.get("/api/jobs/{job_id}/audio")
    def get_audio(job_id: str) -> FileResponse:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job no encontrado: {job_id}")
        if job.state is not JobState.DONE or not job.output_path or not job.output_path.exists():
            raise HTTPException(status_code=409, detail="El audio aún no está disponible.")
        return FileResponse(job.output_path, media_type="audio/wav", filename=job.output_path.name)

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict:
        result = manager.delete(job_id)
        if result == "not_found":
            raise HTTPException(status_code=404, detail=f"Job no encontrado: {job_id}")
        if result == "running":
            raise HTTPException(status_code=409, detail="No se puede eliminar un job en ejecución.")
        return {"deleted": job_id}

    return app


app = create_app()
