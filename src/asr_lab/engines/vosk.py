import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

from .base import ASREngine
from ..core.models import EngineConfig, TranscriptionResult
from ..audio.loader import load_audio


class VoskEngine(ASREngine):
    """ASR engine for Vosk (offline speech recognition)."""

    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.model_path = getattr(self.config, "model_path", None)
        if not self.model_path:
            raise ValueError("VoskEngine requires 'model_path'")
        
        self.model = None
        self.recognizer = None

    def load_model(self) -> None:
        """Loads Vosk model."""
        if self.model is None:
            try:
                from vosk import Model, KaldiRecognizer
            except ImportError:
                raise ImportError("Vosk not installed. Install with: pip install vosk")
            
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer

    def transcribe(self, audio_path: Path, language: str) -> TranscriptionResult:
        """Transcribes audio using Vosk."""
        if self.model is None:
            self.load_model()

        start_time = time.time()
        
        # Load audio (Vosk typically works best with 16kHz)
        audio_data, sr = load_audio(audio_path, target_sr=16000)
        
        # Convert float32 to int16 PCM
        audio_int16 = (audio_data * 32768).astype(np.int16)
        
        rec = self.recognizer(self.model, sr)
        rec.SetWords(True)
        
        results = []
        
        # Process in chunks (e.g. 4000 samples)
        chunk_size = 4000
        for i in range(0, len(audio_int16), chunk_size):
            data = audio_int16[i:i+chunk_size].tobytes()
            if rec.AcceptWaveform(data):
                results.append(json.loads(rec.Result()))
        
        results.append(json.loads(rec.FinalResult()))
        
        elapsed = time.time() - start_time
        text = " ".join([r.get("text", "") for r in results]).strip()

        return TranscriptionResult(
            text=text,
            processing_time=elapsed,
            confidence=None
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Returns model metadata."""
        return {
            "model_path": self.model_path,
            "framework": "Vosk"
        }
