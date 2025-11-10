import time
import json
from pathlib import Path
from typing import Dict, Any
import wave

from .base import ASREngine


class VoskEngine(ASREngine):
    """ASR engine for Vosk (offline speech recognition)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_path = self.config.get("model_path")
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

    def transcribe(self, audio_path: Path, language: str) -> Dict[str, Any]:
        """Transcribes audio using Vosk."""
        if self.model is None:
            self.load_model()

        start_time = time.time()
        
        wf = wave.open(str(audio_path), "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 32000, 48000]:
            raise ValueError("Audio must be WAV mono PCM")
        
        rec = self.recognizer(self.model, wf.getframerate())
        rec.SetWords(True)
        
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                results.append(json.loads(rec.Result()))
        
        results.append(json.loads(rec.FinalResult()))
        wf.close()
        
        elapsed = time.time() - start_time
        text = " ".join([r.get("text", "") for r in results]).strip()

        return {
            "text": text,
            "processing_time": elapsed,
            "confidence": None
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Returns model metadata."""
        return {
            "model_path": self.model_path,
            "framework": "Vosk"
        }
