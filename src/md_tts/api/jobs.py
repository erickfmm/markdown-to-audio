"""Gestor de trabajos con un único worker en segundo plano.

Los modelos TTS son pesados y (a menudo) viven en GPU, por lo que los trabajos
se procesan uno a la vez en orden de llegada (FIFO). El estado de los trabajos
activos vive en memoria, pero los terminados persisten como carpetas en disco
(``<jobs_dir>/<job_id>/<nombre>.wav``): al arrancar (y en cada listado) se
re-escanean para mostrar las conversiones anteriores. No hay base de datos:
la carpeta ``jobs_dir`` es la fuente de verdad.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from .. import service

logger = logging.getLogger("md_tts.api")


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class JobOptions:
    engine: str = "mms"
    language: str = "es"
    pause_ms: int = 500
    device: Optional[str] = None
    workers: int = 1
    save_fragments: bool = False
    qwen_model: Optional[str] = None
    qwen_speaker: Optional[str] = None
    qwen_instruct: Optional[str] = None
    qwen_ref_audio: Optional[str] = None  # ruta resuelta desde voice_ref_id
    qwen_ref_text: Optional[str] = None
    qwen_xvector_only: bool = False

    @classmethod
    def from_values(
        cls,
        engine: str = "mms",
        language: str = "es",
        pause_ms: int = 500,
        device: Optional[str] = None,
        workers: int = 1,
        save_fragments: bool = False,
        qwen_model: Optional[str] = None,
        qwen_speaker: Optional[str] = None,
        qwen_instruct: Optional[str] = None,
        qwen_ref_audio: Optional[str] = None,
        qwen_ref_text: Optional[str] = None,
        qwen_xvector_only: bool = False,
    ) -> "JobOptions":
        try:
            return cls(
                engine=str(engine),
                language=str(language),
                pause_ms=max(0, int(pause_ms)),
                device=(str(device) if device else None),
                workers=min(16, max(1, int(workers))),
                save_fragments=save_fragments in (True, "true", "1", "on", "yes"),
                qwen_model=(str(qwen_model) if qwen_model else None),
                qwen_speaker=(str(qwen_speaker) if qwen_speaker else None),
                qwen_instruct=(str(qwen_instruct) if qwen_instruct else None),
                qwen_ref_audio=(str(qwen_ref_audio) if qwen_ref_audio else None),
                qwen_ref_text=(str(qwen_ref_text) if qwen_ref_text else None),
                qwen_xvector_only=qwen_xvector_only in (True, "true", "1", "on", "yes"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Opción inválida: {exc}") from exc


@dataclass
class Job:
    id: str
    name: str
    content: str
    options: JobOptions
    state: JobState = JobState.QUEUED
    done: int = 0
    total: int = 0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    output_path: Optional[Path] = None
    #: True para jobs históricos descubiertos escaneando jobs_dir (no tienen
    #: contenido ni opciones en memoria).
    from_disk: bool = False

    def to_dict(self) -> dict:
        duration = None
        if self.started_at and self.finished_at:
            duration = round(self.finished_at - self.started_at, 1)
        data = {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "done": self.done,
            "total": self.total,
            "error": self.error,
            "created_at": self.created_at,
            "duration": duration,
            "from_disk": self.from_disk,
            "options": {
                "engine": self.options.engine,
                "language": self.options.language,
                "pause_ms": self.options.pause_ms,
                "device": self.options.device,
                "workers": self.options.workers,
                "save_fragments": self.options.save_fragments,
                "qwen_model": self.options.qwen_model,
                "qwen_speaker": self.options.qwen_speaker,
                "qwen_instruct": self.options.qwen_instruct,
                "qwen_ref_text": self.options.qwen_ref_text,
                "qwen_xvector_only": self.options.qwen_xvector_only,
            }
            if self.options is not None
            else None,
        }
        if self.state is JobState.DONE and self.output_path is not None:
            data["audio_url"] = f"/api/jobs/{self.id}/audio"
            data["mp3_url"] = f"/api/jobs/{self.id}/audio.mp3"
            data["filename"] = self.output_path.name
        return data


class JobManager:
    """Cola FIFO de trabajos procesada por un hilo worker daemon.

    Los trabajos terminados persisten en ``<jobs_dir>/<job_id>/``; además de
    la cola en memoria, el gestor re-escanea el disco para listar, consultar y
    eliminar conversiones de ejecuciones anteriores.
    """

    def __init__(self, jobs_dir: Path) -> None:
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._worker = threading.Thread(target=self._run, name="md-tts-job-worker", daemon=True)
        self._worker.start()

    # -- API pública -------------------------------------------------

    def submit(self, name: str, content: str, options: JobOptions) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], name=name, content=content, options=options)
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put(job.id)
        logger.info("Job %s encolado: %s (engine=%s)", job.id, name, options.engine)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        """Retorna el job activo de memoria o el histórico descubierto en disco."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job
        return self._disk_job(job_id)

    def list_jobs(self) -> List[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        # Fusiona con los históricos de disco; si un id está en memoria
        # (p. ej. re-ejecutando), gana el estado en memoria.
        known_ids = {job.id for job in jobs}
        for disk_id, disk_job in self._scan_disk_jobs().items():
            if disk_id not in known_ids:
                jobs.append(disk_job)
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def delete(self, job_id: str) -> Optional[str]:
        """Elimina un job y sus archivos. Retorna None si OK o el motivo del rechazo."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                job = self._disk_job(job_id)
                if job is None:
                    return "not_found"
            elif job.state is JobState.RUNNING:
                return "running"
            else:
                del self._jobs[job_id]

        job_dir = self._job_dir(job_id)
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
        return None

    # -- Escaneo de disco --------------------------------------------

    def _job_dir(self, job_id: str) -> Path:
        """Ruta de la carpeta del job, validada para que no escape de jobs_dir."""
        clean = str(job_id).strip()
        if not clean or clean in (".", "..") or "/" in clean or "\\" in clean:
            raise ValueError(f"Id de job inválido: {job_id}")
        return self.jobs_dir / clean

    def _disk_job(self, job_id: str) -> Optional[Job]:
        """Construye un Job sintético 'done' desde la carpeta de un job en disco."""
        try:
            job_dir = self._job_dir(job_id)
        except ValueError:
            return None
        if not job_dir.is_dir():
            return None
        wav = _find_output_wav(job_dir)
        if wav is None:
            return None
        return _job_from_wav(job_id, wav)

    def _scan_disk_jobs(self) -> Dict[str, Job]:
        """Escanea jobs_dir y retorna {job_id: Job} por cada WAV de salida."""
        found: Dict[str, Job] = {}
        try:
            entries = list(self.jobs_dir.iterdir())
        except OSError as exc:
            logger.warning("No se pudo escanear %s: %s", self.jobs_dir, exc)
            return found
        for entry in entries:
            if not entry.is_dir():
                continue
            wav = _find_output_wav(entry)
            if wav is not None:
                found[entry.name] = _job_from_wav(entry.name, wav)
        return found

    # -- Worker ------------------------------------------------------

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            with self._lock:
                job = self._jobs.get(job_id)
            if job is None:
                continue  # fue eliminado mientras estaba en cola

            self._update(job, state=JobState.RUNNING, started_at=time.time())
            logger.info("Job %s iniciado: %s", job.id, job.name)

            def report(done: int, total: int, _job: Job = job) -> None:
                self._update(_job, done=done, total=total)

            try:
                output_path = service.process_markdown_text(
                    name=job.name,
                    content=job.content,
                    output_dir=self.jobs_dir / job.id,
                    engine_name=job.options.engine,
                    language=job.options.language,
                    pause_ms=job.options.pause_ms,
                    device=job.options.device,
                    workers=job.options.workers,
                    save_fragments=job.options.save_fragments,
                    qwen_model=job.options.qwen_model,
                    qwen_speaker=job.options.qwen_speaker,
                    qwen_instruct=job.options.qwen_instruct,
                    qwen_ref_audio=job.options.qwen_ref_audio,
                    qwen_ref_text=job.options.qwen_ref_text,
                    qwen_xvector_only=job.options.qwen_xvector_only,
                    progress_callback=report,
                    show_progress=False,
                )
                self._update(job, state=JobState.DONE, output_path=output_path, finished_at=time.time())
                logger.info("Job %s completado: %s", job.id, output_path)
            except Exception as exc:  # noqa: BLE001 - el worker nunca debe morir
                logger.exception("Job %s falló", job.id)
                self._update(job, state=JobState.ERROR, error=str(exc), finished_at=time.time())

    def _update(self, job: Job, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(job, key, value)


# -- Descubrimiento de jobs en disco ----------------------------------


def _find_output_wav(job_dir: Path) -> Optional[Path]:
    """Retorna el WAV de salida de la carpeta de un job (ignora fragments/)."""

    try:
        for entry in job_dir.iterdir():
            if entry.is_file() and entry.suffix.lower() == ".wav":
                return entry
    except OSError as exc:
        logger.warning("No se pudo inspeccionar %s: %s", job_dir, exc)
    return None


def _job_from_wav(job_id: str, wav: Path) -> Job:
    """Construye un Job sintético 'done' a partir de un WAV en disco."""

    try:
        mtime = wav.stat().st_mtime
    except OSError:
        mtime = time.time()
    return Job(
        id=job_id,
        name=wav.stem,
        content="",
        options=None,  # type: ignore[arg-type]
        state=JobState.DONE,
        output_path=wav,
        created_at=mtime,
        finished_at=mtime,
        from_disk=True,
    )
