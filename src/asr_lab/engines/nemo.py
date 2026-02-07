import time 

import torch
from pathlib import Path
from typing import Dict, Any

from .base import ASREngine
from ..core.models import EngineConfig, TranscriptionResult


class NeMoEngine(ASREngine):
    """
    An ASR engine implementation for NVIDIA NeMo models.
    Supports models like Parakeet, QuartzNet, Citrinet, Conformer-CTC.
    """

    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.model_name = getattr(self.config, "model_name", None)
        if not self.model_name:
            raise ValueError("NeMoEngine config must include a 'model_name'")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load_model(self) -> None:
        """Loads the NeMo ASR model."""
        if self.model is None:
            try:
                import nemo.collections.asr as nemo_asr
            except ImportError:
                raise ImportError(
                    "NeMo toolkit not installed. Install with: pip install nemo_toolkit[asr]"
                )
            
            # Load pre-trained model from NVIDIA NGC
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name
            )
            self.model = self.model.to(self.device)
            self.model.eval()

    def transcribe(self, audio_path: Path, language: str) -> TranscriptionResult:
        """Transcribes an audio file using the loaded NeMo model."""
        if self.model is None:
            self.load_model()

        start_time = time.time()
        
        # NeMo transcribe expects a list of file paths
        transcriptions = self.model.transcribe([str(audio_path)])
        
        elapsed = time.time() - start_time

        # Extract the transcription text - handle various return formats
        text = ""
        if transcriptions:
            result = transcriptions[0]
            # Handle Hypothesis objects with text attribute
            if hasattr(result, 'text'):
                text = result.text
            # Handle nested list (some models return [[text]])
            elif isinstance(result, list) and result:
                text = result[0] if isinstance(result[0], str) else str(result[0])
            # Handle direct string
            elif isinstance(result, str):
                text = result
            else:
                text = str(result)

        return TranscriptionResult(
            text=text,
            processing_time=elapsed,
            confidence=None
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the NeMo model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "framework": "NVIDIA NeMo"
        }
