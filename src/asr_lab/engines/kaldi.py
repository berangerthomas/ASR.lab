import time
from pathlib import Path
from typing import Dict, Any
import subprocess

from .base import ASREngine
from ..core.models import EngineConfig, TranscriptionResult


class KaldiEngine(ASREngine):
    """ASR engine for Kaldi (requires external Kaldi installation)."""

    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.model_dir = getattr(self.config, "model_dir", None)
        self.script_path = getattr(self.config, "script_path", None)
        if not self.model_dir or not self.script_path:
            raise ValueError("KaldiEngine requires 'model_dir' and 'script_path'")

    def load_model(self) -> None:
        """Kaldi models are loaded by external scripts."""
        pass

    def transcribe(self, audio_path: Path, language: str) -> TranscriptionResult:
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

        return TranscriptionResult(
            text=text,
            processing_time=elapsed,
            confidence=None
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Returns model metadata."""
        return {
            "model_dir": self.model_dir,
            "framework": "Kaldi"
        }
