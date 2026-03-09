from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple
from multiprocessing import Pool
from functools import partial

from pydub import AudioSegment
from tqdm import tqdm

from . import parser as md_parser
from .audio_utils import concatenate_with_pause
from .tts import build_engine

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("md_tts")
LANGUAGE_AWARE_ENGINES = {"mms", "kokoro", "chatterbox", "cosyvoice"}


def process_paragraph(
    para: md_parser.Paragraph,
    engine_name: str,
    language: str,
    device: str | None,
    cosyvoice_model_dir: Path | None,
    cosyvoice_prompt_wav: Path | None,
    temp_dir: Path | None,
) -> Tuple[int, AudioSegment | None]:
    """Procesa un párrafo individual. Retorna (index, segment)."""
    text = para.text.strip()
    if not text:
        return (para.index, None)
    
    # Verificar si el fragmento ya existe
    if temp_dir:
        fragment_name = f"{para.index:04d}_{para.section[:30] if para.section else 'intro'}.wav"
        fragment_path = temp_dir / fragment_name
        if fragment_path.exists():
            try:
                seg = AudioSegment.from_wav(fragment_path)
                return (para.index, seg)
            except Exception as exc:
                logger.warning("Error cargando fragmento existente %s: %s", fragment_path, exc)
    
    # Sintetizar el fragmento
    try:
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
            fragment_name = f"{para.index:04d}_{para.section[:30] if para.section else 'intro'}.wav"
            fragment_path = temp_dir / fragment_name
            seg.export(fragment_path, format="wav")
        
        return (para.index, seg)
    except Exception as exc:
        logger.error("Error sintetizando párrafo %s: %s", para.index, exc)
        return (para.index, None)


def process_markdown_file(
    md_file: Path,
    output_dir: Path,
    engine_name: str,
    language: str,
    pause_ms: int,
    device: str | None,
    cosyvoice_model_dir: Path | None = None,
    cosyvoice_prompt_wav: Path | None = None,
    save_fragments: bool = False,
    workers: int = 1,
) -> Path:
    paragraphs = md_parser.split_markdown(md_file)
    logger.info("%s -> %d fragmentos", md_file.name, len(paragraphs))

    # Crear carpeta para fragmentos si se solicita
    temp_dir = None
    if save_fragments:
        temp_dir = output_dir / "fragments" / md_file.stem
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Contar fragmentos existentes
        existing_fragments = list(temp_dir.glob("*.wav"))
        if existing_fragments:
            logger.info("Encontrados %d fragmentos existentes, reanudando...", len(existing_fragments))

    segments: List[Tuple[int, AudioSegment]] = []
    
    if workers > 1:
        # Procesamiento paralelo
        process_func = partial(
            process_paragraph,
            engine_name=engine_name,
            language=language,
            device=device,
            cosyvoice_model_dir=cosyvoice_model_dir,
            cosyvoice_prompt_wav=cosyvoice_prompt_wav,
            temp_dir=temp_dir,
        )
        
        with Pool(processes=workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, paragraphs),
                desc=f"{md_file.name}",
                unit="p",
                total=len(paragraphs)
            ))
        
        segments = [(idx, seg) for idx, seg in results if seg is not None]
    else:
        # Procesamiento secuencial
        for para in tqdm(paragraphs, desc=f"{md_file.name}", unit="p"):
            idx, seg = process_paragraph(
                para,
                engine_name=engine_name,
                language=language,
                device=device,
                cosyvoice_model_dir=cosyvoice_model_dir,
                cosyvoice_prompt_wav=cosyvoice_prompt_wav,
                temp_dir=temp_dir,
            )
            if seg is not None:
                segments.append((idx, seg))
    
    if not segments:
        raise RuntimeError(f"No se generaron segmentos de audio para {md_file}")
    
    # Ordenar segmentos por índice para mantener el orden correcto
    segments.sort(key=lambda x: x[0])
    audio_segments = [seg for _, seg in segments]

    combined = concatenate_with_pause(audio_segments, pause_ms=pause_ms)
    output_path = output_dir / f"{md_file.stem}.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path, format="wav")
    return output_path


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
