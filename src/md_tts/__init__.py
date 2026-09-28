"""Herramientas para convertir documentos Markdown en audio usando TTS.

Incluye el CLI (``md-tts``), el servidor web (``md-tts-web``: GUI Flask +
backend FastAPI) y la lógica de servicio reutilizable.
"""

__all__ = [
    "cli",
    "parser",
    "tts",
    "audio_utils",
    "service",
    "api",
    "web",
    "server_cli",
]
