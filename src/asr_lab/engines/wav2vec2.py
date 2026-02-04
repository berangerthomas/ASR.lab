import time 

import torch
from pathlib import Path
from typing import Dict, Any
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa

from .base import ASREngine
from ..core.models import EngineConfig, TranscriptionResult


class Wav2Vec2Engine(ASREngine):
    """ASR engine for Meta's Wav2Vec2 models."""

    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.model_id = getattr(self.config, "model_id", None) or self.config.model_name
        if not self.model_id:
            raise ValueError("Wav2Vec2Engine requires 'model_id' or 'model_name'")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        """Loads Wav2Vec2 model and processor."""
        if self.model is None:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id).to(self.device)
            self.model.eval()

    def transcribe(self, audio_path: Path, language: str) -> TranscriptionResult:
        """Transcribes audio using Wav2Vec2."""
        if self.model is None:
            self.load_model()

        start_time = time.time()
        
        audio_data, sampling_rate = librosa.load(audio_path, sr=16000)
        inputs = self.processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        elapsed = time.time() - start_time

        return TranscriptionResult(
            text=text,
            processing_time=elapsed,
            confidence=None
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Returns model metadata."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "framework": "Wav2Vec2"
        }
