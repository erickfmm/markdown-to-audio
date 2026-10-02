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
        choices=[
            "mms",
            "kokoro",
            "chatterbox",
            "vibevoice",
            "cosyvoice",
            "qwen3-customvoice",
            "qwen3-voicedesign",
            "qwen3-clone",
        ],
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
        help="cpu, cuda o auto (motores qwen3 usan GPU si está disponible)",
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
        "--qwen-model",
        type=str,
        default=None,
        help="Motor qwen3: tamaño '1.7b'/'0.6b' o un id de HuggingFace/ruta local.",
    )
    parser.add_argument(
        "--qwen-speaker",
        type=str,
        default=None,
        help="qwen3-customvoice: voz premium (Vivian, Serena, Uncle_Fu, Dylan, "
        "Eric, Ryan, Aiden, Ono_Anna, Sohee).",
    )
    parser.add_argument(
        "--qwen-instruct",
        type=str,
        default=None,
        help="qwen3: instrucción de estilo (customvoice 1.7b) o descripción de "
        "la voz a diseñar (voicedesign, obligatoria).",
    )
    parser.add_argument(
        "--qwen-ref-audio",
        type=Path,
        default=None,
        help="qwen3-clone: audio de referencia para clonar la voz (wav/mp3, "
        "ideal 3-10 s de una sola voz).",
    )
    parser.add_argument(
        "--qwen-ref-text",
        type=str,
        default=None,
        help="qwen3-clone: transcripción del audio de referencia (mejora el clon).",
    )
    parser.add_argument(
        "--qwen-xvector-only",
        action="store_true",
        help="qwen3-clone: clonar solo con el embedding del hablante (sin transcripción).",
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
    _validate_qwen_args(args)

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
            qwen_model=args.qwen_model,
            qwen_speaker=args.qwen_speaker,
            qwen_instruct=args.qwen_instruct,
            qwen_ref_audio=str(args.qwen_ref_audio) if args.qwen_ref_audio else None,
            qwen_ref_text=args.qwen_ref_text,
            qwen_xvector_only=args.qwen_xvector_only,
            save_fragments=args.save_fragments,
            workers=args.workers,
        )
        logger.info("Audio generado: %s", out_path)
        if args.save_fragments:
            logger.info("Fragmentos guardados en: %s/fragments/%s", output_dir, md_file.stem)


def _validate_qwen_args(args: argparse.Namespace) -> None:
    """Valida las opciones qwen3-* antes de sintetizar (errores claros)."""

    from .tts import QWEN_SPEAKERS

    if args.engine == "qwen3-voicedesign" and not (args.qwen_instruct or "").strip():
        raise SystemExit(
            "qwen3-voicedesign requiere --qwen-instruct con la descripción de la voz "
            "(género, edad, timbre, emoción, ritmo)."
        )
    if args.engine == "qwen3-clone":
        if not args.qwen_ref_audio:
            raise SystemExit("qwen3-clone requiere --qwen-ref-audio (audio de referencia para clonar).")
        if not args.qwen_ref_audio.is_file():
            raise SystemExit(f"No existe el audio de referencia: {args.qwen_ref_audio}")
        if not (args.qwen_ref_text or "").strip() and not args.qwen_xvector_only:
            raise SystemExit(
                "qwen3-clone requiere --qwen-ref-text (transcripción) o --qwen-xvector-only."
            )
    if args.engine == "qwen3-customvoice":
        if args.qwen_speaker and args.qwen_speaker not in QWEN_SPEAKERS:
            raise SystemExit(
                f"Speaker desconocido: {args.qwen_speaker}. Opciones: {', '.join(QWEN_SPEAKERS)}"
            )
        if (args.qwen_instruct or "").strip() and str(args.qwen_model or "1.7b").lower() == "0.6b":
            raise SystemExit("El modelo 0.6B-CustomVoice no soporta --qwen-instruct; use 1.7b.")


if __name__ == "__main__":
    main()
