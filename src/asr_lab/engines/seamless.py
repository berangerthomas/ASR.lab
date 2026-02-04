import time
import torch
import soundfile as sf
from pathlib import Path
from typing import Dict, Any

from .base import ASREngine
from ..core.models import EngineConfig, TranscriptionResult


class SeamlessM4TEngine(ASREngine):
    """ASR engine for Meta's SeamlessM4T (multilingual speech translation)."""
    
    # Language code mapping: ISO 639-1 (2-letter) -> ISO 639-3 (3-letter) used by SeamlessM4T
    LANGUAGE_MAP = {
        "fr": "fra",
        "en": "eng",
        "es": "spa",
        "de": "deu",
        "it": "ita",
        "pt": "por",
        "nl": "nld",
        "pl": "pol",
        "ru": "rus",
        "zh": "cmn",
        "ja": "jpn",
        "ko": "kor",
        "ar": "arb",
        "hi": "hin",
        "tr": "tur",
        "vi": "vie",
        "th": "tha",
        "cs": "ces",
        "da": "dan",
        "fi": "fin",
        "el": "ell",
        "he": "heb",
        "hu": "hun",
        "id": "ind",
        "no": "nob",
        "ro": "ron",
        "sv": "swe",
        "uk": "ukr",
    }

    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.model_name = self.config.model_name or "facebook/seamless-m4t-large"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        """Loads SeamlessM4T model."""
        if self.model is None:
            try:
                from transformers import SeamlessM4TModel, AutoProcessor
                import torchaudio
            except ImportError:
                raise ImportError("transformers>=4.34.0 and torchaudio required for SeamlessM4T")
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = SeamlessM4TModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.torchaudio = torchaudio

    def transcribe(self, audio_path: Path, language: str) -> TranscriptionResult:
        """Transcribes audio using SeamlessM4T."""
        if self.model is None:
            self.load_model()

        start_time = time.time()
        
        # Convert language code to SeamlessM4T format (ISO 639-3)
        tgt_lang = self.LANGUAGE_MAP.get(language.lower(), language)
        
        # Use soundfile instead of torchaudio.load to avoid TorchCodec dependency issues on Windows
        waveform, sample_rate = sf.read(str(audio_path))
        waveform = torch.from_numpy(waveform).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
        audio_input = waveform

        if sample_rate != 16000:
            resampler = self.torchaudio.transforms.Resample(sample_rate, 16000)
            audio_input = resampler(audio_input)
        
        inputs = self.processor(audio=audio_input.squeeze().numpy(), return_tensors="pt", sampling_rate=16000)
        
        with torch.no_grad():
            output_tokens = self.model.generate(**inputs.to(self.device), tgt_lang=tgt_lang, generate_speech=False)
        
        text = self.processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        
        elapsed = time.time() - start_time

        return TranscriptionResult(
            text=text,
            processing_time=elapsed,
            confidence=None
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Returns model metadata."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "framework": "SeamlessM4T"
        }
