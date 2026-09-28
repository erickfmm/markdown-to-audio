from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from . import parser as md_parser
from .service import LANGUAGE_AWARE_ENGINES, process_markdown_file

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("md_tts")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convierte Markdown a audio (es/en según motor).")
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "modificacion1",
        help="Directorio con archivos .md",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "output_audio",
        help="Directorio donde guardar los .wav",
    )
    parser.add_argument(
        "--engine",
        choices=["mms", "kokoro", "chatterbox", "vibevoice", "cosyvoice"],
        default="mms",
        help="Motor TTS a usar",
    )
    parser.add_argument(
        "--language",
        choices=["en", "es"],
        default="es",
        help="Idioma del TTS cuando el motor lo soporte (default: es)",
    )
    parser.add_argument(
        "--pause-ms",
        type=int,
        default=500,
        help="Pausa (ms) entre párrafos",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu o cuda (si disponible)",
    )
    parser.add_argument(
        "--cosyvoice-model-dir",
        type=Path,
        default=None,
        help="Ruta local del modelo CosyVoice (opcional; si no, descarga).",
    )
    parser.add_argument(
        "--cosyvoice-prompt-wav",
        type=Path,
        default=None,
        help="WAV de referencia de voz para CosyVoice (opcional).",
    )
    parser.add_argument(
        "--save-fragments",
        action="store_true",
        help="Guardar fragmentos individuales en output_dir/fragments/ (permite reanudar)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Número de workers paralelos para procesamiento (default: 1)",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de entrada: {input_dir}")

    if args.engine not in LANGUAGE_AWARE_ENGINES and args.language != "es":
        logger.warning(
            "El engine '%s' no usa --language actualmente; se ignorará '%s'.",
            args.engine,
            args.language,
        )
    if args.engine == "vibevoice" and args.language == "es":
        logger.warning(
            "VibeVoice está optimizado principalmente para inglés; el resultado en español puede degradarse."
        )

    for md_file in md_parser.iter_markdown_files(input_dir):
        out_path = process_markdown_file(
            md_file=md_file,
            output_dir=output_dir,
            engine_name=args.engine,
            language=args.language,
            pause_ms=args.pause_ms,
            device=args.device,
            cosyvoice_model_dir=args.cosyvoice_model_dir,
            cosyvoice_prompt_wav=args.cosyvoice_prompt_wav,
            save_fragments=args.save_fragments,
            workers=args.workers,
        )
        logger.info("Audio generado: %s", out_path)
        if args.save_fragments:
            logger.info("Fragmentos guardados en: %s/fragments/%s", output_dir, md_file.stem)


if __name__ == "__main__":
    main()
