from __future__ import annotations

import logging
import inspect
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pydub import AudioSegment

from .audio_utils import merge_numpy, numpy_to_segment

logger = logging.getLogger(__name__)


class TtsEngine:
    name: str

    def synthesize(self, text: str) -> AudioSegment:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class MmsEngine(TtsEngine):
    language: str = "es"
    device: str = "cpu"

    def __post_init__(self):
        from transformers import AutoTokenizer, VitsModel
        import torch

        model_by_language = {
            "es": "facebook/mms-tts-spa",
            "en": "facebook/mms-tts-eng",
        }
        if self.language not in model_by_language:
            raise ValueError(f"Idioma no soportado para MMS: {self.language}. Use 'es' o 'en'.")

        model_id = model_by_language[self.language]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = VitsModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.name = model_id

    def synthesize(self, text: str) -> AudioSegment:
        import torch

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model(**inputs).waveform.squeeze(0).cpu().numpy()
        return numpy_to_segment(output, sample_rate=self.model.config.sampling_rate)


@dataclass
class KokoroEngine(TtsEngine):
    language: str = "es"
    device: Optional[str] = None

    def __post_init__(self):
        from kokoro import KPipeline

        if self.language not in {"es", "en"}:
            raise ValueError(f"Idioma no soportado para Kokoro: {self.language}. Use 'es' o 'en'.")

        self.name = "hexgrad/Kokoro-82M"
        self.pipeline = KPipeline(lang_code=self.language, repo_id=self.name, device=self.device or "cpu")

    def synthesize(self, text: str) -> AudioSegment:
        chunks = []
        for _, _, audio in self.pipeline(text, voice="em_alex"):
            chunks.append(audio)
        merged = merge_numpy([np.array(c, dtype=np.float32) for c in chunks])
        if merged.size == 0:
            raise RuntimeError("No se produjo audio con Kokoro")
        return numpy_to_segment(merged, sample_rate=24000)


@dataclass
class ChatterboxEngine(TtsEngine):
    language: str = "es"
    device: Optional[str] = None

    def __post_init__(self):
        if self.language not in {"es", "en"}:
            raise ValueError(f"Idioma no soportado para Chatterbox: {self.language}. Use 'es' o 'en'.")

        self.multilingual = False
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device or "cpu")
            self.multilingual = True
            self.name = "ResembleAI/chatterbox-multilingual"
        except Exception:
            from chatterbox.tts import ChatterboxTTS

            # Fallback para versiones sin soporte multilingüe explícito.
            self.model = ChatterboxTTS.from_pretrained(device=self.device or "cpu")
            self.name = "ResembleAI/chatterbox"

    def synthesize(self, text: str) -> AudioSegment:
        if self.multilingual:
            wav = self.model.generate(text, language_id=self.language)
        else:
            wav = self.model.generate(text)
        data = wav.cpu().numpy() if hasattr(wav, "cpu") else np.asarray(wav)
        return numpy_to_segment(data.squeeze(), sample_rate=self.model.sr)


@dataclass
class VibeVoiceEngine(TtsEngine):
    device: Optional[str] = None

    def __post_init__(self):
        from transformers import pipeline

        device_arg = -1
        if self.device and self.device != "cpu":
            device_arg = 0 if self.device in ("cuda", "cuda:0") else self.device

        self.pipe = pipeline(
            "text-to-speech",
            model="microsoft/VibeVoice-Realtime-0.5B",
            device=device_arg,
        )
        self.name = "microsoft/VibeVoice-Realtime-0.5B"

    def synthesize(self, text: str) -> AudioSegment:
        result = self.pipe(text)
        audio = result["audio"] if isinstance(result, dict) else result
        sr = result.get("sampling_rate", 24000) if isinstance(result, dict) else 24000
        return numpy_to_segment(np.asarray(audio, dtype=np.float32), sample_rate=sr)


@dataclass
class CosyVoiceEngine(TtsEngine):
    language: str = "es"
    device: Optional[str] = None
    model_dir: Optional[str] = None
    prompt_wav: Optional[str] = None

    def __post_init__(self):
        try:
            from cosyvoice.cli.cosyvoice import AutoModel
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - depende de extras
            raise ImportError(
                "CosyVoice no está instalado. Instale desde https://github.com/FunAudioLLM/CosyVoice "
                "y asegure las dependencias (torch, torchaudio, funasr)."
            ) from exc

        local_dir = self.model_dir or snapshot_download(
            "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir="/tmp/cosyvoice3"
        )
        self.model = AutoModel(model_dir=local_dir, device=self.device or "cpu")
        self.prompt_wav = self.prompt_wav or None
        self.name = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

    def synthesize(self, text: str) -> AudioSegment:
        if self.language not in {"es", "en"}:
            raise ValueError(f"Idioma no soportado para CosyVoice: {self.language}. Use 'es' o 'en'.")

        prompt_text_by_language = {
            "en": "You are a helpful assistant.<|endofprompt|>Please read naturally in English.",
            "es": "You are a helpful assistant.<|endofprompt|>Por favor lee de forma natural en español.",
        }
        prompt_text = prompt_text_by_language[self.language]

        prompt_wav = self.prompt_wav
        chunks = []
        for _, out in enumerate(
            self.model.inference_zero_shot(
                text,
                prompt_text,
                prompt_wav,
                stream=False,
            )
        ):
            wav = out.get("tts_speech")
            if wav is not None:
                chunks.append(wav.squeeze())
        if not chunks:
            raise RuntimeError("CosyVoice no devolvió audio")
        merged = merge_numpy([np.array(c, dtype=np.float32) for c in chunks])
        return numpy_to_segment(merged, sample_rate=self.model.sample_rate)


ENGINE_REGISTRY = {
    "mms": MmsEngine,
    "kokoro": KokoroEngine,
    "chatterbox": ChatterboxEngine,
    "vibevoice": VibeVoiceEngine,
    "cosyvoice": CosyVoiceEngine,
}


def build_engine(name: str, **kwargs) -> TtsEngine:
    key = name.lower()
    if key not in ENGINE_REGISTRY:
        raise KeyError(f"Engine desconocido: {name}. Opciones: {list(ENGINE_REGISTRY)}")
    engine_cls = ENGINE_REGISTRY[key]
    sig = inspect.signature(engine_cls)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return engine_cls(**filtered)
