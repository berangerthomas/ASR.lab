import time 

import torch
from pathlib import Path
from typing import Dict, Any

from .base import ASREngine


class NeMoEngine(ASREngine):
    """
    An ASR engine implementation for NVIDIA NeMo models.
    Supports models like Parakeet, QuartzNet, Citrinet, Conformer-CTC.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = self.config.get("model_name")
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

    def transcribe(self, audio_path: Path, language: str) -> Dict[str, Any]:
        """Transcribes an audio file using the loaded NeMo model."""
        if self.model is None:
            self.load_model()

        start_time = time.time()
        
        # NeMo transcribe expects a list of file paths
        transcriptions = self.model.transcribe([str(audio_path)])
        
        elapsed = time.time() - start_time

        # Extract the transcription text - handle Hypothesis objects
        if transcriptions and hasattr(transcriptions[0], 'text'):
            # If it's a Hypothesis object with a text attribute
            text = transcriptions[0].text
        elif transcriptions:
            # If it's already a string
            text = transcriptions[0]
        else:
            text = ""

        return {
            "text": text,
            "processing_time": elapsed,
            "confidence": None  # NeMo doesn't provide confidence by default
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the NeMo model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "framework": "NVIDIA NeMo"
        }
