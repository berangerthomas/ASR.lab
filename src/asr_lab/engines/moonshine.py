import time
import logging
from typing import Dict, Any, Optional
import torch
import librosa
from pathlib import Path
import transformers

from .base import ASREngine
from ..core.models import EngineConfig, TranscriptionResult

logger = logging.getLogger(__name__)


class MoonshineEngine(ASREngine):
    """
    ASR engine for Useful Sensors Moonshine models.

    Moonshine is a family of speech recognition models optimized for
    real-time on-device inference with a low memory footprint.
    Available models: moonshine-base, moonshine-tiny.

    Uses the Hugging Face Transformers pipeline for inference.
    Moonshine supports inputs up to ~64 seconds natively.
    """

    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.model_id = getattr(self.config, "model_id", None) or self.config.model_name
        if not self.model_id:
            raise ValueError(
                "MoonshineEngine config must include a 'model_id' or 'model_name'. "
                "Example: 'UsefulSensors/moonshine-base'"
            )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pipe: Optional[Any] = None

    def load_model(self) -> None:
        """Loads the Moonshine model via the Transformers pipeline."""
        if self.pipe is None:
            logger.info(f"Loading Moonshine model: {self.model_id}")
            
            # Retrieve chunk_length_s and stride_length_s from the YAML engine config.
            # E.g., chunk_length_s: 30 in moonshine.yaml.
            # Default to None if not specified.
            chunk_length_s = getattr(self.config, "chunk_length_s", None)
            stride_length_s = getattr(self.config, "stride_length_s", None)
            
            pipeline_kwargs = {
                "task": "automatic-speech-recognition",
                "model": self.model_id,
                "device": self.device,
            }
            if chunk_length_s is not None:
                pipeline_kwargs["chunk_length_s"] = chunk_length_s
                # Moonshine cannot export timestamps, so HF pipeline text overlap deduplication fails.
                # Setting stride_length_s to [0, 0] prevents the pipeline from repeating overlapping audio text.
                pipeline_kwargs["stride_length_s"] = stride_length_s if stride_length_s is not None else [0, 0]

            self.pipe = transformers.pipeline(**pipeline_kwargs)
            logger.info(f"Moonshine model loaded on {self.device}")

    def unload_model(self) -> None:
        """Unloads the model to free memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transcribe(self, audio_path: Path, language: str) -> TranscriptionResult:
        """
        Transcribes an audio file using the Moonshine model.

        Moonshine is English-only and does not support language selection;
        the `language` parameter is accepted for API compatibility but ignored.
        """
        if self.pipe is None:
            self.load_model()

        start_time = time.time()
        try:
            # Load audio at 16 kHz (required by Moonshine)
            import numpy as np
            audio, _ = librosa.load(str(audio_path), sr=16000)
            
            # Ajout d'une seconde de silence (padding) au début et à la fin de l'audio
            # pour éviter que Moonshine ne tronque les extrémités par manque de contexte.
            silence = np.zeros(16000, dtype=np.float32)
            padded_audio = np.concatenate((silence, audio, silence))

            result = self.pipe({"sampling_rate": 16000, "raw": padded_audio})
            text = str(result["text"]).strip()  # type: ignore[index]
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            text = ""

        elapsed = time.time() - start_time
        return TranscriptionResult(
            text=text,
            processing_time=elapsed,
            confidence=None,
            metadata={"model": self.model_id},
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the Moonshine model."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "language": "en",
        }