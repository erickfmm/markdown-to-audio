"""App Flask que sirve la GUI y actúa de proxy transparente hacia FastAPI.

IMPORTANTE: este módulo debe depender únicamente de ``flask`` y ``requests``
para permitir imágenes Docker ligeras del frontend (sin torch/transformers).
La URL del backend se toma de ``MD_TTS_API_URL`` o del parámetro ``api_url``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests
from flask import Flask, Response, render_template, request

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_STATIC_DIR = Path(__file__).resolve().parent / "static"

_DEFAULT_API_URL = "http://127.0.0.1:8001"

# Cabeceras que no deben reenviarse tal cual entre proxies.
_SKIP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-encoding",
    "content-length",
}


def create_app(api_url: Optional[str] = None) -> Flask:
    app = Flask(__name__, template_folder=str(_TEMPLATE_DIR), static_folder=str(_STATIC_DIR))
    app.config["MD_TTS_API_URL"] = (
        api_url or os.environ.get("MD_TTS_API_URL") or _DEFAULT_API_URL
    ).rstrip("/")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/<path:path>", methods=["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS"])
    def api_proxy(path: str):
        target = f"{app.config['MD_TTS_API_URL']}/api/{path}"
        headers = {}
        if "Content-Type" in request.headers:
            headers["Content-Type"] = request.headers["Content-Type"]
        body = None if request.method in ("GET", "HEAD") else request.get_data()

        try:
            upstream = requests.request(
                method=request.method,
                url=target,
                params=request.args,
                data=body,
                headers=headers,
                stream=True,
                timeout=(10, 300),
            )
        except requests.RequestException as exc:
            return {"error": f"Backend no disponible: {exc.__class__.__name__}"}, 502

        passthrough = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in _SKIP_HEADERS
        }
        return Response(
            upstream.iter_content(chunk_size=64 * 1024),
            status=upstream.status_code,
            headers=passthrough,
        )

    return app


app = create_app()
