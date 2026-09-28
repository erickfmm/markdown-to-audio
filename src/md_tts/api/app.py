"""App FastAPI con los endpoints de trabajos y metadata del servicio."""

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
from ..tts import ENGINE_REGISTRY
from .jobs import JobManager, JobOptions, JobState

logger = logging.getLogger("md_tts.api")

DEFAULT_JOBS_DIR = Path(os.environ.get("MD_TTS_JOBS_DIR", "output_audio/web"))
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class JobCreateRequest(BaseModel):
    name: str = Field(default="documento", min_length=1, max_length=120)
    content: str = Field(min_length=1)
    engine: str = "mms"
    language: str = "es"
    pause_ms: int = Field(default=500, ge=0, le=10000)
    device: Optional[str] = None
    workers: int = Field(default=1, ge=1, le=16)
    save_fragments: bool = False


def _parse_options(
    engine: str,
    language: str,
    pause_ms,
    device,
    workers,
    save_fragments,
) -> JobOptions:
    if engine not in ENGINE_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Engine desconocido: {engine}. Opciones: {list(ENGINE_REGISTRY)}",
        )
    if language not in ("es", "en"):
        raise HTTPException(status_code=400, detail=f"Idioma inválido: {language}. Use 'es' o 'en'.")
    if device not in (None, "", "cpu", "cuda"):
        raise HTTPException(status_code=400, detail=f"Device inválido: {device}. Use 'cpu' o 'cuda'.")
    try:
        return JobOptions.from_values(
            engine=engine,
            language=language,
            pause_ms=pause_ms,
            device=device,
            workers=workers,
            save_fragments=save_fragments,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def create_app(jobs_dir: Optional[Path] = None) -> FastAPI:
    app = FastAPI(
        title="md-tts",
        description="API para convertir Markdown a audio mediante trabajos asíncronos.",
        version="0.1.0",
    )
    manager = JobManager(jobs_dir or DEFAULT_JOBS_DIR)
    app.state.jobs = manager

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/engines")
    def engines() -> dict:
        return {
            "engines": [
                {"id": name, "language_aware": name in service.LANGUAGE_AWARE_ENGINES}
                for name in ENGINE_REGISTRY
            ],
            "languages": ["es", "en"],
        }

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
