"""CLI del servidor web: GUI Flask + backend FastAPI embebido.

Uso:

    md-tts-web --host 0.0.0.0 --port 5000

Inicia el backend FastAPI en un hilo (puerto ``--port + 1`` por defecto) y la
GUI Flask en el hilo principal. Con ``--api-url`` se ejecuta en modo solo-GUI
contra un backend externo (útil para contenedores ligeros del frontend).
"""

from __future__ import annotations

import argparse
import logging
import socket
import threading
import time
from pathlib import Path
from typing import List, Optional

from .web.app import create_app as create_flask_app

logger = logging.getLogger("md_tts.web")


def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="md-tts-web",
        description="Lanza la GUI web (Flask) junto al backend (FastAPI).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host para la GUI Flask (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Puerto para la GUI Flask (default: 5000)")
    parser.add_argument(
        "--api-host",
        default="127.0.0.1",
        help="Host del backend FastAPI embebido (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=None,
        help="Puerto del backend FastAPI embebido (default: --port + 1)",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help=(
            "URL de un backend FastAPI externo (p. ej. http://127.0.0.1:8000). "
            "Si se define, no se inicia el backend embebido."
        ),
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=None,
        help="Directorio de salida del audio del backend (default: output_audio/web)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_arg_parser().parse_args(argv)

    api_port = args.api_port if args.api_port is not None else args.port + 1
    server = None

    if args.api_url:
        api_url = args.api_url.rstrip("/")
        logger.info("Modo solo-GUI: backend externo en %s", api_url)
    else:
        # Import perezoso: permite despliegues ligeros del frontend sin fastapi/uvicorn.
        import uvicorn

        from .api.app import create_app as create_api_app

        api_url = f"http://{args.api_host}:{api_port}"
        api_kwargs = {"jobs_dir": args.jobs_dir}
        if args.jobs_dir:
            # Deriva el almacén de voces del --jobs-dir explícito; si no,
            # create_app usa sus defaults (env MD_TTS_* o rutas relativas).
            api_kwargs["voice_refs_dir"] = Path(args.jobs_dir).parent / "voice_refs"
        api_app = create_api_app(**api_kwargs)
        server = uvicorn.Server(
            uvicorn.Config(api_app, host=args.api_host, port=api_port, log_level="warning")
        )
        thread = threading.Thread(target=server.run, daemon=True, name="md-tts-api")
        thread.start()
        if _wait_for_port(args.api_host, api_port, timeout=30):
            logger.info("Backend FastAPI listo en %s", api_url)
        else:
            logger.warning("El backend no abrió el puerto %s (revisa los logs).", api_port)

    flask_app = create_flask_app(api_url=api_url)
    logger.info("GUI en http://%s:%s (Ctrl+C para detener)", args.host, args.port)
    try:
        flask_app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
    finally:
        if server is not None:
            server.should_exit = True


if __name__ == "__main__":
    main()
