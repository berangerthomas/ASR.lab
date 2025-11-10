import time
from pathlib import Path
from typing import Dict, Any
import subprocess

from .base import ASREngine


class KaldiEngine(ASREngine):
    """ASR engine for Kaldi (requires external Kaldi installation)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_dir = self.config.get("model_dir")
        self.script_path = self.config.get("script_path")
        if not self.model_dir or not self.script_path:
            raise ValueError("KaldiEngine requires 'model_dir' and 'script_path'")

    def load_model(self) -> None:
        """Kaldi models are loaded by external scripts."""
        pass

    def transcribe(self, audio_path: Path, language: str) -> Dict[str, Any]:
        """Transcribes audio using Kaldi via subprocess."""
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [self.script_path, str(audio_path), self.model_dir],
                capture_output=True,
                text=True,
                timeout=300
            )
            text = result.stdout.strip()
        except subprocess.TimeoutExpired:
            text = ""
        except Exception as e:
            raise RuntimeError(f"Kaldi transcription failed: {e}")
        
        elapsed = time.time() - start_time

        return {
            "text": text,
            "processing_time": elapsed,
            "confidence": None
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Returns model metadata."""
        return {
            "model_dir": self.model_dir,
            "framework": "Kaldi"
        }
