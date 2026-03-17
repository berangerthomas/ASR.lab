import time
import logging
import torch
from pathlib import Path
from typing import Dict, Any

from .base import ASREngine
from ..core.models import EngineConfig, TranscriptionResult

logger = logging.getLogger(__name__)


class SenseVoiceEngine(ASREngine):
    """
    ASR engine for Alibaba SenseVoice models via FunASR.

    SenseVoice is a speech foundation model with capabilities for multilingual
    speech recognition, emotion recognition, and audio event detection.
    Supports automatic language detection.

    Available models:
      - FunAudioLLM/SenseVoiceSmall (recommended, fast)
      - iic/SenseVoiceSmall (alternative mirror)

    Dependencies: `funasr` and `modelscope` (included in core dependencies).
    """

    # Language codes supported by SenseVoice
    SUPPORTED_LANGUAGES = {
        "auto", "zh", "en", "ja", "ko", "yue",
    }

    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.model_id = (
            getattr(self.config, "model_id", None)
            or self.config.model_name
            or "iic/SenseVoiceSmall"
        )
        # Whether to apply Inverse Text Normalization (numbers/punctuation)
        self.use_itn: bool = getattr(self.config, "use_itn", True)
        # Device string expected by FunASR ("cuda:0" or "cpu")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load_model(self) -> None:
        """Loads the SenseVoice model via FunASR."""
        if self.model is not None:
            return
        from funasr import AutoModel  # noqa: PLC0415

        logger.info(f"Loading SenseVoice model: {self.model_id}")
        self.model = AutoModel(
            model=self.model_id,
            trust_remote_code=True,
            device=self.device,
            disable_pbar=True,
        )
        logger.info(f"SenseVoice model loaded on {self.device}")

    def unload_model(self) -> None:
        """Unloads the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transcribe(self, audio_path: Path, language: str) -> TranscriptionResult:
        """
        Transcribes an audio file using SenseVoice.

        Args:
            audio_path: Path to the audio file.
            language: BCP-47 or SenseVoice language code. Passing an empty
                      string or None defaults to automatic detection ("auto").
        """
        if self.model is None:
            self.load_model()

        # Map language to a SenseVoice-supported code; fall back to auto
        lang = language if language in self.SUPPORTED_LANGUAGES else "auto"

        start_time = time.time()
        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess  # noqa: PLC0415  # noqa: PLC0415

            res = self.model.generate(
                input=str(audio_path),
                cache={},
                language=lang,
                use_itn=self.use_itn,
                batch_size_s=60,
            )
            raw_text = res[0]["text"] if res else ""
            text = rich_transcription_postprocess(raw_text).strip()
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            text = ""

        elapsed = time.time() - start_time
        return TranscriptionResult(
            text=text,
            processing_time=elapsed,
            confidence=None,
            metadata={"model": self.model_id, "language_detected": lang},
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the SenseVoice model."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "use_itn": self.use_itn,
            "supported_languages": sorted(self.SUPPORTED_LANGUAGES),
        }