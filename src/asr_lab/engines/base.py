from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path

from ..core.models import EngineConfig, TranscriptionResult

class ASREngine(ABC):
    """Abstract base class for all ASR engines"""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.name = config.id
        
    @abstractmethod
    def load_model(self) -> None:
        """Loads the ASR model"""
        pass
    
    @abstractmethod
    def transcribe(self, audio_path: Path, language: str) -> TranscriptionResult:
        """
        Transcribes an audio file.

        Returns:
            A TranscriptionResult object containing text, confidence, and processing time.
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata about the model"""
        pass

