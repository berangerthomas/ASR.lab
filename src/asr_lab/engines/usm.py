import time
from pathlib import Path
from typing import Dict, Any

from .base import ASREngine


class USMEngine(ASREngine):
    """
    ASR engine for Google's Universal Speech Model (USM).
    Note: USM is not publicly available as open-source.
    This is a placeholder for future API integration.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = self.config.get("api_key")
        raise NotImplementedError(
            "USM (Universal Speech Model) is not publicly available. "
            "Use Google Cloud Speech-to-Text API instead (see engines/api/google.py)"
        )

    def load_model(self) -> None:
        """USM not available for local deployment."""
        pass

    def transcribe(self, audio_path: Path, language: str) -> Dict[str, Any]:
        """Not implemented."""
        raise NotImplementedError("USM not available")

    def get_metadata(self) -> Dict[str, Any]:
        """Returns placeholder metadata."""
        return {
            "framework": "USM (Google)",
            "status": "Not publicly available"
        }
