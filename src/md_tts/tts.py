from __future__ import annotations

import logging
import inspect
from dataclasses import dataclass
from pathlib import Path
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
    cosyvoice_model_dir: Optional[str] = None
    cosyvoice_prompt_wav: Optional[str] = None

    def __post_init__(self):
        try:
            from cosyvoice.cli.cosyvoice import AutoModel
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - depende de extras
            raise ImportError(
                "CosyVoice no está instalado. Instale desde https://github.com/FunAudioLLM/CosyVoice "
                "y asegure las dependencias (torch, torchaudio, funasr)."
            ) from exc

        local_dir = self.cosyvoice_model_dir or snapshot_download(
            "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir="/tmp/cosyvoice3"
        )
        self.model = AutoModel(model_dir=local_dir, device=self.device or "cpu")
        self.cosyvoice_prompt_wav = self.cosyvoice_prompt_wav or None
        self.name = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

    def synthesize(self, text: str) -> AudioSegment:
        if self.language not in {"es", "en"}:
            raise ValueError(f"Idioma no soportado para CosyVoice: {self.language}. Use 'es' o 'en'.")

        prompt_text_by_language = {
            "en": "You are a helpful assistant.<|endofprompt|>Please read naturally in English.",
            "es": "You are a helpful assistant.<|endofprompt|>Por favor lee de forma natural en español.",
        }
        prompt_text = prompt_text_by_language[self.language]

        prompt_wav = self.cosyvoice_prompt_wav
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


# ---------------------------------------------------------------------------
# Qwen3-TTS — https://huggingface.co/collections/Qwen/qwen3-tts
#
# Requiere el paquete `qwen-tts` (extra opcional `md-tts[qwen]`). Los pesos se
# descargan de HuggingFace en la primera ejecución (el codec
# Qwen3-TTS-Tokenizer-12Hz se descarga automáticamente junto al modelo).
# ---------------------------------------------------------------------------

#: Speakers premium de los modelos CustomVoice (1.7B y 0.6B).
QWEN_SPEAKERS = (
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
)

#: Alias corto -> id de HuggingFace por familia de modelos Qwen3-TTS.
QWEN_MODEL_ALIASES = {
    "customvoice": {
        "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    },
    "voicedesign": {
        "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    },
    "base": {
        "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    },
}

#: Mapa idioma interno -> nombre de idioma que espera Qwen3-TTS.
QWEN_LANGUAGE_NAMES = {"es": "Spanish", "en": "English"}


def resolve_qwen_model_id(variant: str, requested: Optional[str]) -> str:
    """Resuelve un alias (``1.7b``/``0.6b``) al id de HuggingFace de la familia.

    Cualquier otro valor se devuelve tal cual, lo que permite usar ids
    completos de HuggingFace (p. ej. fine-tunes) o rutas locales.
    """

    aliases = QWEN_MODEL_ALIASES[variant]
    if not requested:
        return aliases["1.7b"]
    key = str(requested).strip()
    return aliases.get(key.lower(), key)


class Qwen3TtsEngineBase(TtsEngine):
    """Base común de los motores Qwen3-TTS: dispositivo y carga del modelo."""

    variant: str = "customvoice"  # clave en QWEN_MODEL_ALIASES

    def _resolve_device(self) -> str:
        """Normaliza cpu/cuda/auto: ``auto`` usa GPU si está disponible."""

        import torch

        device = (getattr(self, "device", None) or "auto").strip().lower()
        if device in ("", "auto", "none"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA solicitado pero no disponible; usando CPU.")
            device = "cpu"
        return device

    def _load_model(self, model_id: str) -> None:
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:  # pragma: no cover - depende de extras
            raise ImportError(
                "qwen-tts no está instalado. Instale con: pip install 'md-tts[qwen]' "
                "(o pip install qwen-tts). Más info: https://github.com/QwenLM/Qwen3-TTS"
            ) from exc

        import torch

        device = self._resolve_device()
        device_map = "cpu" if device == "cpu" else ("cuda:0" if device == "cuda" else device)
        if device_map.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        logger.info("Cargando %s en %s (dtype=%s)", model_id, device_map, str(dtype))
        self.model = Qwen3TTSModel.from_pretrained(model_id, device_map=device_map, dtype=dtype)
        self.name = model_id
        self.qwen_model_id = model_id

    def _qwen_language(self) -> str:
        return QWEN_LANGUAGE_NAMES.get((getattr(self, "language", "") or "es").lower(), "Auto")

    def _to_segment(self, wavs, sample_rate: int) -> AudioSegment:
        audio = np.asarray(wavs[0], dtype=np.float32) if len(wavs) else np.array([], dtype=np.float32)
        if audio.size == 0:
            raise RuntimeError("Qwen3-TTS no devolvió audio")
        return numpy_to_segment(audio, sample_rate=sample_rate)


@dataclass
class Qwen3CustomVoiceEngine(Qwen3TtsEngineBase):
    """Qwen3-TTS-12Hz-{1.7B,0.6B}-CustomVoice: 9 voces premium + instruct (solo 1.7B)."""

    variant: str = "customvoice"
    language: str = "es"
    device: Optional[str] = None
    qwen_model: Optional[str] = None
    qwen_speaker: Optional[str] = None
    qwen_instruct: Optional[str] = None

    def __post_init__(self):
        model_id = resolve_qwen_model_id("customvoice", self.qwen_model)
        self._load_model(model_id)

        speaker = (self.qwen_speaker or "Vivian").strip()
        if speaker not in QWEN_SPEAKERS:
            raise ValueError(f"Speaker desconocido: {speaker}. Opciones: {list(QWEN_SPEAKERS)}")
        self.qwen_speaker = speaker
        if (self.qwen_instruct or "").strip() and "0.6B" in model_id:
            logger.warning("instruct solo está soportado por 1.7B-CustomVoice; se ignorará.")
            self.qwen_instruct = None

    def synthesize(self, text: str) -> AudioSegment:
        kwargs = {
            "text": text,
            "language": self._qwen_language(),
            "speaker": self.qwen_speaker,
        }
        if (self.qwen_instruct or "").strip():
            kwargs["instruct"] = self.qwen_instruct.strip()
        wavs, sr = self.model.generate_custom_voice(**kwargs)
        return self._to_segment(wavs, sr)


@dataclass
class Qwen3VoiceDesignEngine(Qwen3TtsEngineBase):
    """Qwen3-TTS-12Hz-1.7B-VoiceDesign: diseña la voz desde una descripción."""

    variant: str = "voicedesign"
    language: str = "es"
    device: Optional[str] = None
    qwen_model: Optional[str] = None
    qwen_instruct: Optional[str] = None

    def __post_init__(self):
        if not (self.qwen_instruct or "").strip():
            raise ValueError(
                "qwen3-voicedesign requiere una descripción de la voz (qwen_instruct). "
                "Ej.: 'Voz femenina adulta, timbre cálido, tono sereno de narradora.'"
            )
        model_id = resolve_qwen_model_id("voicedesign", self.qwen_model)
        self._load_model(model_id)

    def synthesize(self, text: str) -> AudioSegment:
        wavs, sr = self.model.generate_voice_design(
            text=text,
            language=self._qwen_language(),
            instruct=(self.qwen_instruct or "").strip(),
        )
        return self._to_segment(wavs, sr)


@dataclass
class Qwen3VoiceCloneEngine(Qwen3TtsEngineBase):
    """Qwen3-TTS-12Hz-{1.7B,0.6B}-Base: clonación de voz desde audio de referencia."""

    variant: str = "base"
    language: str = "es"
    device: Optional[str] = None
    qwen_model: Optional[str] = None
    qwen_ref_audio: Optional[str] = None
    qwen_ref_text: Optional[str] = None
    qwen_xvector_only: bool = False

    def __post_init__(self):
        if not self.qwen_ref_audio:
            raise ValueError("qwen3-clone requiere un audio de referencia (qwen_ref_audio).")
        self._validate_ref()
        model_id = resolve_qwen_model_id("base", self.qwen_model)
        self._load_model(model_id)
        # Prompts de clonación cacheados por ruta de audio: cambiar la
        # referencia no recarga el modelo, solo construye otro prompt.
        self._clone_prompts: dict = {}

    def _validate_ref(self) -> None:
        if not Path(self.qwen_ref_audio).is_file():
            raise FileNotFoundError(f"No existe el audio de referencia: {self.qwen_ref_audio}")
        if not (self.qwen_ref_text or "").strip() and not self.qwen_xvector_only:
            raise ValueError(
                "qwen3-clone requiere la transcripción del audio (qwen_ref_text) o el modo "
                "x-vector-only (clona solo con el embedding del hablante, menor calidad)."
            )

    def _clone_prompt(self):
        # La referencia puede cambiar entre jobs sobre un engine cacheado.
        # La clave incluye modo/transcripción: cambiarlos construye otro prompt.
        self._validate_ref()
        key = (
            str(Path(self.qwen_ref_audio).resolve()),
            bool(self.qwen_xvector_only),
            (self.qwen_ref_text or "").strip(),
        )
        prompt = self._clone_prompts.get(key)
        if prompt is None:
            prompt = self.model.create_voice_clone_prompt(
                ref_audio=key[0],
                ref_text=key[2] or None,
                x_vector_only_mode=key[1],
            )
            self._clone_prompts[key] = prompt
        return prompt

    def synthesize(self, text: str) -> AudioSegment:
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=self._qwen_language(),
            voice_clone_prompt=self._clone_prompt(),
        )
        return self._to_segment(wavs, sr)


ENGINE_REGISTRY = {
    "mms": MmsEngine,
    "kokoro": KokoroEngine,
    "chatterbox": ChatterboxEngine,
    "vibevoice": VibeVoiceEngine,
    "cosyvoice": CosyVoiceEngine,
    "qwen3-customvoice": Qwen3CustomVoiceEngine,
    "qwen3-voicedesign": Qwen3VoiceDesignEngine,
    "qwen3-clone": Qwen3VoiceCloneEngine,
}

#: Metadatos por engine para la API/UI (validación y opciones dinámicas).
ENGINE_METADATA = {
    "qwen3-customvoice": {
        "variants": ["1.7b", "0.6b"],
        "default_variant": "1.7b",
        "speakers": list(QWEN_SPEAKERS),
        "default_speaker": "Vivian",
        "supports_instruct": True,
        "requires_instruct": False,
        "supports_voice_clone": False,
        "requires_voice_reference": False,
    },
    "qwen3-voicedesign": {
        "variants": ["1.7b"],
        "default_variant": "1.7b",
        "speakers": [],
        "default_speaker": None,
        "supports_instruct": True,
        "requires_instruct": True,
        "supports_voice_clone": False,
        "requires_voice_reference": False,
    },
    "qwen3-clone": {
        "variants": ["1.7b", "0.6b"],
        "default_variant": "1.7b",
        "speakers": [],
        "default_speaker": None,
        "supports_instruct": False,
        "requires_instruct": False,
        "supports_voice_clone": True,
        "requires_voice_reference": True,
    },
}


def build_engine(name: str, **kwargs) -> TtsEngine:
    key = name.lower()
    if key not in ENGINE_REGISTRY:
        raise KeyError(f"Engine desconocido: {name}. Opciones: {list(ENGINE_REGISTRY)}")
    engine_cls = ENGINE_REGISTRY[key]
    sig = inspect.signature(engine_cls)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return engine_cls(**filtered)
