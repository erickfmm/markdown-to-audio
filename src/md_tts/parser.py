from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class Paragraph:
    """Representa un fragmento de texto listo para sintetizar."""

    text: str
    section: str
    index: int


HEADING_PATTERN = re.compile(r"^(#+)\s+(.*)$")


def _flush_buffer(
    paragraphs: List[Paragraph], buffer: List[str], section: str, idx_counter: List[int]
) -> None:
    if not buffer:
        return
    text = " ".join(line.strip() for line in buffer).strip()
    if text:
        paragraphs.append(Paragraph(text=text, section=section, index=idx_counter[0]))
        idx_counter[0] += 1
    buffer.clear()


def split_markdown_text(content: str) -> List[Paragraph]:
    """Divide el contenido Markdown (texto) en párrafos por secciones (#) y bloques vacíos.

    - Cada encabezado se incluye como un párrafo independiente (para anunciar la sección).
    - Los párrafos se separan por líneas en blanco.
    """

    paragraphs: List[Paragraph] = []
    buffer: List[str] = []
    current_section = ""
    idx_counter = [0]

    for line in content.splitlines():
        heading_match = HEADING_PATTERN.match(line.strip())
        if heading_match:
            _flush_buffer(paragraphs, buffer, current_section, idx_counter)
            current_section = heading_match.group(2).strip()
            # Anunciamos la sección como un párrafo corto
            paragraphs.append(
                Paragraph(
                    text=current_section,
                    section=current_section,
                    index=idx_counter[0],
                )
            )
            idx_counter[0] += 1
            continue

        if not line.strip():
            _flush_buffer(paragraphs, buffer, current_section, idx_counter)
            continue

        buffer.append(line.rstrip())

    _flush_buffer(paragraphs, buffer, current_section, idx_counter)
    return paragraphs


def split_markdown(file_path: Path) -> List[Paragraph]:
    """Lee un archivo Markdown y lo divide en párrafos (delega en split_markdown_text)."""

    content = file_path.read_text(encoding="utf-8")
    return split_markdown_text(content)


def iter_markdown_files(root: Path) -> Iterable[Path]:
    """Devuelve los archivos markdown en el directorio (ordenados)."""

    for path in sorted(root.glob("*.md")):
        if path.is_file():
            yield path
